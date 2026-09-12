const http = require("node:http");
const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const path = require("node:path");
const { URL } = require("node:url");

const ROOT = path.resolve(__dirname);
const PUBLIC_DIR = path.join(ROOT, "public");
const DEFAULT_DATA_DIR = path.resolve(process.env.BRAND_VERIFICATION_DATA_DIR || path.join(ROOT, "data", "annotations"));
const DEFAULT_MANIFEST_PATH = path.resolve(process.env.BRAND_VERIFICATION_MANIFEST || path.join(ROOT, "data", "example_manifest.json"));
const DEFAULT_IMAGE_DIR = path.resolve(process.env.BRAND_VERIFICATION_IMAGE_DIR || path.join(ROOT, "data", "images"));
const DEFAULT_PORT = Number(process.env.PORT || 5182);
const HOST = String(process.env.HOST || "127.0.0.1");
const ANNOTATION_SCHEMA_VERSION = "brand_verification_annotation_v1";

function contentType(filePath) {
  return {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml"
  }[path.extname(filePath).toLowerCase()] || "application/octet-stream";
}

function sendJson(res, status, value, extraHeaders = {}) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
    ...extraHeaders
  });
  res.end(body);
}

function sendText(res, status, value, type = "text/plain; charset=utf-8") {
  res.writeHead(status, { "content-type": type, "content-length": Buffer.byteLength(value) });
  res.end(value);
}

async function readBody(req, limit = 2 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw Object.assign(new Error("Request body is too large."), { statusCode: 413 });
    chunks.push(chunk);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

function validSessionId(value) {
  const id = String(value || "");
  return /^\d{8}$/.test(id) ? id : null;
}

function safeItemFilename(itemId) {
  return `${String(itemId).replace(/[^a-zA-Z0-9_.-]+/g, "_")}.json`;
}

function isWithin(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function resolveImagePath(image, imageDir) {
  const filename = path.basename(String(image?.filename || ""));
  if (!filename) throw Object.assign(new Error("Manifest image has no filename."), { statusCode: 400 });
  const resolved = path.resolve(imageDir, filename);
  if (!isWithin(imageDir, resolved)) throw Object.assign(new Error("Image path escapes the configured image directory."), { statusCode: 403 });
  return resolved;
}

function flattenItems(manifest) {
  return manifest.images.flatMap((image) => image.advertisements.map((ad) => ({
    ...ad,
    image_id: image.image_id,
    filename: image.filename,
    image_position: image.image_position,
    image_metadata: image.metadata || {},
    page_type: image.page_type || "single"
  })));
}

function completionErrors(annotation, categories) {
  const errors = [];
  if (annotation.identification_wrong) return errors;
  if (!["correct", "incorrect"].includes(annotation.category_review)) {
    errors.push("Mark the assigned brand category as correct or incorrect.");
  }
  if (annotation.category_review === "incorrect" && !categories.includes(annotation.corrected_brand_category)) {
    errors.push("Choose the correct brand category.");
  }
  if (!["correct", "incorrect"].includes(annotation.brand_name_review)) {
    errors.push("Mark the assigned brand name as correct or incorrect.");
  }
  if (annotation.brand_name_review === "incorrect" && !String(annotation.corrected_brand_name || "").trim()) {
    errors.push("Enter the correct brand name, or enter “unclear”.");
  }
  return errors;
}

function normalizeAnnotation(item, input, categories, completeRequested = false, existing = null) {
  const now = new Date().toISOString();
  const identificationWrong = Boolean(input.identification_wrong);
  const categoryReview = ["correct", "incorrect"].includes(input.category_review) ? input.category_review : null;
  const brandReview = ["correct", "incorrect"].includes(input.brand_name_review) ? input.brand_name_review : null;
  const correctedCategory = categoryReview === "incorrect" && categories.includes(input.corrected_brand_category)
    ? input.corrected_brand_category
    : null;
  const correctedBrand = brandReview === "incorrect" ? String(input.corrected_brand_name || "").trim() || null : null;
  const output = {
    schema_version: ANNOTATION_SCHEMA_VERSION,
    item_id: item.item_id,
    image_id: item.image_id,
    filename: item.filename,
    advertisement_id: item.advertisement_id,
    item_position: item.item_position,
    original: {
      bbox_1000: item.bbox_1000,
      face_boxes_1000: item.face_boxes_1000 || [],
      brand_category: item.assigned_brand_category,
      brand_name: item.assigned_brand_name
    },
    identification_wrong: identificationWrong,
    identification_note: String(input.identification_note || "").trim(),
    no_faces: Boolean(input.no_faces),
    category_review: categoryReview,
    corrected_brand_category: correctedCategory,
    effective_brand_category: identificationWrong ? null : (categoryReview === "correct" ? item.assigned_brand_category : correctedCategory),
    brand_name_review: brandReview,
    corrected_brand_name: correctedBrand,
    effective_brand_name: identificationWrong ? null : (brandReview === "correct" ? item.assigned_brand_name : correctedBrand),
    notes: String(input.notes || "").trim(),
    status: "draft",
    created_at: existing?.created_at || now,
    updated_at: now,
    completed_at: null
  };
  const errors = completionErrors(output, categories);
  if (completeRequested && errors.length) {
    const error = new Error(errors.join(" "));
    error.statusCode = 400;
    throw error;
  }
  if (completeRequested) {
    output.status = "complete";
    output.completed_at = now;
  } else if (existing?.status === "complete" && errors.length === 0) {
    output.status = "complete";
    output.completed_at = existing.completed_at || now;
  }
  return output;
}

async function atomicWriteJson(filePath, value) {
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await fs.rename(temporary, filePath);
}

async function createApplication(options = {}) {
  const dataDir = path.resolve(options.dataDir || DEFAULT_DATA_DIR);
  const manifestPath = path.resolve(options.manifestPath || DEFAULT_MANIFEST_PATH);
  const imageDir = path.resolve(options.imageDir || DEFAULT_IMAGE_DIR);
  const manifest = JSON.parse((await fs.readFile(manifestPath, "utf8")).replace(/^\uFEFF/, ""));
  const items = flattenItems(manifest);
  const itemById = new Map(items.map((item) => [item.item_id, item]));
  const imageById = new Map(manifest.images.map((image) => [image.image_id, image]));
  await fs.mkdir(dataDir, { recursive: true });

  function sessionDir(sessionId) {
    const valid = validSessionId(sessionId);
    if (!valid) throw Object.assign(new Error("A valid eight-digit session ID is required."), { statusCode: 400 });
    return path.join(dataDir, valid);
  }

  function annotationPath(sessionId, itemId) {
    if (!itemById.has(itemId)) throw Object.assign(new Error("Unknown verification item."), { statusCode: 404 });
    return path.join(sessionDir(sessionId), safeItemFilename(itemId));
  }

  async function readJson(filePath, fallback = null) {
    try {
      return JSON.parse((await fs.readFile(filePath, "utf8")).replace(/^\uFEFF/, ""));
    } catch (error) {
      if (error.code === "ENOENT") return fallback;
      throw error;
    }
  }

  async function requireSession(sessionId) {
    const session = await readJson(path.join(sessionDir(sessionId), "session.json"));
    if (!session) throw Object.assign(new Error("Saved session not found."), { statusCode: 404 });
    return session;
  }

  async function createSession() {
    for (let attempt = 0; attempt < 25; attempt += 1) {
      const id = String(crypto.randomInt(10_000_000, 100_000_000));
      const dir = path.join(dataDir, id);
      try {
        await fs.mkdir(dir);
      } catch (error) {
        if (error.code === "EEXIST") continue;
        throw error;
      }
      const session = {
        schema_version: "brand_verification_session_v1",
        session_id: id,
        session_code: id,
        task_id: manifest.task_id,
        created_at: new Date().toISOString()
      };
      await atomicWriteJson(path.join(dir, "session.json"), session);
      return session;
    }
    throw new Error("Could not allocate a session number.");
  }

  async function statusMap(sessionId) {
    await requireSession(sessionId);
    const statuses = {};
    await Promise.all(items.map(async (item) => {
      const saved = await readJson(annotationPath(sessionId, item.item_id));
      statuses[item.item_id] = saved?.status === "complete" ? "complete" : (saved ? "started" : "not_started");
    }));
    return statuses;
  }

  async function wrongIdentificationMap(sessionId) {
    await requireSession(sessionId);
    const values = {};
    await Promise.all(items.map(async (item) => {
      const saved = await readJson(annotationPath(sessionId, item.item_id));
      values[item.item_id] = Boolean(saved?.identification_wrong);
    }));
    return values;
  }

  async function listSessions() {
    const entries = await fs.readdir(dataDir, { withFileTypes: true });
    const sessions = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !validSessionId(entry.name)) continue;
      const session = await readJson(path.join(dataDir, entry.name, "session.json"));
      if (!session || session.task_id !== manifest.task_id) continue;
      const statuses = await statusMap(entry.name);
      const complete = Object.values(statuses).filter((value) => value === "complete").length;
      const started = Object.values(statuses).filter((value) => value === "started").length;
      let updatedAt = session.created_at;
      try {
        const stat = await fs.stat(path.join(dataDir, entry.name, "annotations.jsonl"));
        updatedAt = stat.mtime.toISOString();
      } catch {}
      sessions.push({ ...session, complete, started, total: items.length, updated_at: updatedAt });
    }
    return sessions.sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)));
  }

  async function regenerateJsonl(sessionId) {
    const records = [];
    for (const item of items) {
      const saved = await readJson(annotationPath(sessionId, item.item_id));
      if (saved) records.push(saved);
    }
    const body = records.map((record) => JSON.stringify(record)).join("\n") + (records.length ? "\n" : "");
    const target = path.join(sessionDir(sessionId), "annotations.jsonl");
    const temporary = `${target}.${process.pid}.${Date.now()}.tmp`;
    await fs.writeFile(temporary, body, "utf8");
    await fs.rename(temporary, target);
  }

  async function serveStatic(reqPath, res) {
    const relative = reqPath === "/" ? "index.html" : reqPath.replace(/^\/+/, "");
    const filePath = path.resolve(PUBLIC_DIR, relative);
    if (filePath !== PUBLIC_DIR && !filePath.startsWith(`${PUBLIC_DIR}${path.sep}`)) return false;
    try {
      const stat = await fs.stat(filePath);
      if (!stat.isFile()) return false;
      const body = await fs.readFile(filePath);
      res.writeHead(200, { "content-type": contentType(filePath), "content-length": body.length });
      res.end(body);
      return true;
    } catch (error) {
      if (error.code === "ENOENT") return false;
      throw error;
    }
  }

  async function handler(req, res) {
    const parsed = new URL(req.url, "http://localhost");
    const pathname = parsed.pathname;
    try {
      if (req.method === "GET" && pathname === "/health") {
        return sendJson(res, 200, { ok: true, app: "annotation-brand-verification-app" });
      }
      if (req.method === "GET" && pathname === "/api/bootstrap") {
        return sendJson(res, 200, {
          ok: true,
          task: { id: manifest.task_id, name: manifest.name, image_count: manifest.image_count, item_count: items.length },
          brand_categories: manifest.brand_categories,
          sessions: await listSessions()
        });
      }
      if (req.method === "POST" && pathname === "/api/session") {
        const body = await readBody(req);
        if (body.session_id) {
          const session = await requireSession(body.session_id);
          return sendJson(res, 200, { ok: true, resumed: true, session });
        }
        return sendJson(res, 201, { ok: true, resumed: false, session: await createSession() });
      }
      if (req.method === "GET" && pathname === "/api/items") {
        const sessionId = parsed.searchParams.get("session_id");
        const [statuses, wrongIdentifications] = await Promise.all([
          statusMap(sessionId),
          wrongIdentificationMap(sessionId)
        ]);
        return sendJson(res, 200, { ok: true, items, statuses, wrong_identifications: wrongIdentifications });
      }
      if (req.method === "GET" && pathname === "/api/annotation") {
        const sessionId = parsed.searchParams.get("session_id");
        const itemId = parsed.searchParams.get("item_id");
        await requireSession(sessionId);
        const annotation = await readJson(annotationPath(sessionId, itemId));
        return sendJson(res, 200, { ok: true, annotation });
      }
      if (req.method === "POST" && pathname === "/api/annotation") {
        const body = await readBody(req);
        const sessionId = validSessionId(body.session_id);
        await requireSession(sessionId);
        const item = itemById.get(String(body.item_id || ""));
        if (!item) throw Object.assign(new Error("Unknown verification item."), { statusCode: 404 });
        const filePath = annotationPath(sessionId, item.item_id);
        const existing = await readJson(filePath);
        const annotation = normalizeAnnotation(item, body.annotation || {}, manifest.brand_categories, Boolean(body.complete), existing);
        annotation.session = { session_id: sessionId, task_id: manifest.task_id };
        await atomicWriteJson(filePath, annotation);
        await regenerateJsonl(sessionId);
        return sendJson(res, 200, { ok: true, annotation });
      }
      if (req.method === "GET" && pathname === "/api/image") {
        const image = imageById.get(String(parsed.searchParams.get("image_id") || ""));
        if (!image) throw Object.assign(new Error("Unknown image."), { statusCode: 404 });
        const resolved = resolveImagePath(image, imageDir);
        const body = await fs.readFile(resolved);
        res.writeHead(200, {
          "content-type": contentType(resolved),
          "content-length": body.length,
          "cache-control": "private, max-age=3600"
        });
        return res.end(body);
      }
      if (req.method === "GET" && pathname === "/api/export") {
        const sessionId = parsed.searchParams.get("session_id");
        await requireSession(sessionId);
        const file = path.join(sessionDir(sessionId), "annotations.jsonl");
        let body;
        try { body = await fs.readFile(file); } catch (error) { if (error.code === "ENOENT") body = Buffer.from(""); else throw error; }
        res.writeHead(200, {
          "content-type": "application/x-ndjson; charset=utf-8",
          "content-length": body.length,
          "content-disposition": `attachment; filename="${sessionId}_brand_verification.jsonl"`
        });
        return res.end(body);
      }
      if (req.method === "GET" && await serveStatic(pathname, res)) return;
      return sendText(res, 404, "Not found.");
    } catch (error) {
      const status = error.statusCode || (error instanceof SyntaxError ? 400 : 500);
      if (status >= 500) console.error(error);
      return sendJson(res, status, { ok: false, error: error.message || "Unexpected error." });
    }
  }

  return { handler, manifest, items, dataDir, manifestPath, imageDir };
}

async function start() {
  const app = await createApplication();
  const server = http.createServer(app.handler);
  server.listen(DEFAULT_PORT, HOST, () => {
    console.log(`Brand verification app running at http://${HOST}:${DEFAULT_PORT}`);
    console.log(`${app.manifest.image_count} images; ${app.items.length} ad verification items.`);
  });
}

if (require.main === module) {
  start().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

module.exports = {
  ANNOTATION_SCHEMA_VERSION,
  completionErrors,
  createApplication,
  flattenItems,
  normalizeAnnotation,
  resolveImagePath,
  safeItemFilename,
  validSessionId
};
