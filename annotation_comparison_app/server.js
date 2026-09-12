const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");
const url = require("node:url");
const crypto = require("node:crypto");

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, "public");
const DATA_DIR = path.join(ROOT, "data");

function configuredPath(name, fallback) {
  return path.resolve(process.env[name] || fallback);
}

function configuredDirectories(name, fallback) {
  const values = String(process.env[name] || fallback)
    .split(path.delimiter)
    .map((value) => value.trim())
    .filter(Boolean);
  return [...new Set(values.map((value) => path.resolve(value)))];
}

const DEFAULT_MANIFEST = configuredPath("ANNOTATION_COMPARISON_MANIFEST", path.join(DATA_DIR, "example_manifest.json"));
const MATCH_DIR = configuredPath("ANNOTATION_COMPARISON_MATCH_DIR", path.join(DATA_DIR, "manual_matches"));
const LLM_OUTPUT_DIR = configuredPath("ANNOTATION_COMPARISON_LLM_DIR", path.join(DATA_DIR, "llm"));
const HUMAN_SESSIONS_DIR = configuredPath("ANNOTATION_COMPARISON_HUMAN_SESSIONS_DIR", path.join(DATA_DIR, "human_sessions"));
const APP_V2_DATA_DIR = configuredPath("ANNOTATION_COMPARISON_MANIFEST_DIR", path.join(DATA_DIR, "manifests"));
const HUMAN_RESULTS_DIR = configuredPath("ANNOTATION_COMPARISON_HUMAN_RESULTS_DIR", path.join(DATA_DIR, "human_exports"));
const IMAGE_DIR_OPTIONS = configuredDirectories("ANNOTATION_COMPARISON_IMAGE_DIRS", path.join(DATA_DIR, "images"))
  .map((directory, index) => [`Configured image directory ${index + 1}`, directory]);
const IMAGE_SEARCH_DIRS = IMAGE_DIR_OPTIONS.map(([, dir]) => dir);
const DEFAULT_PORT = Number(process.env.PORT || 5177);
const HOST = String(process.env.HOST || "127.0.0.1");
const ALLOWED_IMAGE_PATHS = new Set();

function contentType(filePath) {
  return {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".svg": "image/svg+xml"
  }[path.extname(filePath).toLowerCase()] || "application/octet-stream";
}

function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store"
  });
  res.end(body);
}

function sendText(res, status, value, type = "text/plain; charset=utf-8") {
  res.writeHead(status, { "content-type": type, "content-length": Buffer.byteLength(value) });
  res.end(value);
}

function slug(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "source";
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function isWithin(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function allowedImageRoots(manifestDir, imageDir = "") {
  return [...new Set([
    manifestDir,
    imageDir || null,
    ...IMAGE_SEARCH_DIRS
  ].filter(Boolean).map((value) => path.resolve(value)))];
}

async function readJson(filePath) {
  return JSON.parse((await fs.readFile(filePath, "utf-8")).replace(/^\uFEFF/, ""));
}

async function readRequestJson(req, maxBytes = 1024 * 1024) {
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > maxBytes) {
      const error = new Error("Request body is too large.");
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf-8"));
}

async function readJsonl(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  const records = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      records.push(JSON.parse(line));
    } catch {
      // A partial append must not make the entire source unavailable.
    }
  }
  return records;
}

function isComparableAnnotation(annotation) {
  return Boolean(
    annotation &&
    typeof annotation === "object" &&
    annotation.page &&
    Array.isArray(annotation.advertisements)
  );
}

function filenameStem(value) {
  if (value === undefined || value === null || String(value).trim() === "") return null;
  return path.basename(String(value).trim()).replace(/\.[^.]+$/, "").toLowerCase();
}

function sha1(value) {
  return crypto.createHash("sha1").update(String(value)).digest("hex");
}

function entityRefKey(ref) {
  return [ref?.source_id || "", ref?.ad_id || "", ref?.entity_id || ""].join("\u001f");
}

function normalizeEntityRef(ref) {
  return {
    source_id: String(ref?.source_id || ""),
    ad_id: String(ref?.ad_id || ""),
    entity_id: String(ref?.entity_id || "")
  };
}

function orderedMatchEndpoints(left, right) {
  const normalized = [normalizeEntityRef(left), normalizeEntityRef(right)];
  normalized.sort((a, b) => entityRefKey(a).localeCompare(entityRefKey(b)));
  return normalized;
}

function sourcePairFile(sourceA, sourceB) {
  const sourceIds = [String(sourceA || ""), String(sourceB || "")].sort();
  const hash = sha1(sourceIds.join("\u001f")).slice(0, 12);
  return {
    sourceIds,
    filePath: path.join(MATCH_DIR, `${slug(sourceIds[0])}__${slug(sourceIds[1])}__${hash}.json`)
  };
}

function matchId(imageId, kind, left, right) {
  const [a, b] = orderedMatchEndpoints(left, right);
  return sha1([imageId, kind, entityRefKey(a), entityRefKey(b)].join("\u001e"));
}

async function readMatchFile(filePath, sourceIds) {
  try {
    return await readJson(filePath);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    return {
      schema_version: "annotation_comparison_manual_matches_v1",
      source_ids: sourceIds,
      matches: []
    };
  }
}

async function writeMatchFile(filePath, data) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function readManualMatches(imageId = "") {
  if (!(await pathExists(MATCH_DIR))) return [];
  const entries = await fs.readdir(MATCH_DIR, { withFileTypes: true });
  const matches = [];
  for (const entry of entries.filter((item) => item.isFile() && item.name.endsWith(".json"))) {
    try {
      const parsed = await readJson(path.join(MATCH_DIR, entry.name));
      for (const match of parsed.matches || []) {
        if (!imageId || match.image_id === imageId) matches.push(match);
      }
    } catch {
      // A malformed match file must not prevent comparison from loading.
    }
  }
  return matches;
}

async function saveManualMatch(input) {
  const imageId = String(input?.image_id || "");
  const kind = String(input?.kind || "");
  const [left, right] = orderedMatchEndpoints(input?.left, input?.right);
  if (!imageId || !kind || !left.source_id || !right.source_id || !left.entity_id || !right.entity_id) {
    const error = new Error("image_id, kind, and two entity references are required.");
    error.statusCode = 400;
    throw error;
  }
  if (left.source_id === right.source_id) {
    const error = new Error("Manual matches must connect two different sources.");
    error.statusCode = 400;
    throw error;
  }
  const { sourceIds, filePath } = sourcePairFile(left.source_id, right.source_id);
  const data = await readMatchFile(filePath, sourceIds);
  const id = matchId(imageId, kind, left, right);
  if (!data.matches.some((match) => match.id === id)) {
    data.matches.push({
      id,
      image_id: imageId,
      kind,
      left,
      right,
      created_at: new Date().toISOString()
    });
    data.matches.sort((a, b) => `${a.image_id}:${a.kind}:${a.id}`.localeCompare(`${b.image_id}:${b.kind}:${b.id}`));
    await writeMatchFile(filePath, data);
  }
  return readManualMatches(imageId);
}

async function deleteManualMatch(input) {
  if (input?.id) {
    const imageId = String(input.image_id || "");
    if (!(await pathExists(MATCH_DIR))) return [];
    const entries = await fs.readdir(MATCH_DIR, { withFileTypes: true });
    for (const entry of entries.filter((item) => item.isFile() && item.name.endsWith(".json"))) {
      const filePath = path.join(MATCH_DIR, entry.name);
      const data = await readJson(filePath);
      const before = data.matches?.length || 0;
      data.matches = (data.matches || []).filter((match) => match.id !== input.id);
      if (data.matches.length !== before) await writeMatchFile(filePath, data);
    }
    return readManualMatches(imageId);
  }
  const imageId = String(input?.image_id || "");
  const kind = String(input?.kind || "");
  const [left, right] = orderedMatchEndpoints(input?.left, input?.right);
  const { sourceIds, filePath } = sourcePairFile(left.source_id, right.source_id);
  const data = await readMatchFile(filePath, sourceIds);
  const id = matchId(imageId, kind, left, right);
  const before = data.matches.length;
  data.matches = data.matches.filter((match) => match.id !== id);
  if (data.matches.length !== before) await writeMatchFile(filePath, data);
  return readManualMatches(imageId);
}

async function newestHumanResultManifest() {
  if (!(await pathExists(HUMAN_RESULTS_DIR))) return null;
  const entries = await fs.readdir(HUMAN_RESULTS_DIR, { withFileTypes: true });
  const candidates = [];
  for (const entry of entries.filter((item) => item.isFile() && item.name.endsWith(".json"))) {
    const filePath = path.join(HUMAN_RESULTS_DIR, entry.name);
    try {
      const parsed = await readJson(filePath);
      if (!Array.isArray(parsed?.annotation_set?.manifest?.images)) continue;
      const stat = await fs.stat(filePath);
      candidates.push({
        manifest: parsed.annotation_set.manifest,
        sourcePath: filePath,
        modifiedMs: stat.mtimeMs
      });
    } catch {
      // Ignore result files that are not aggregate exports.
    }
  }
  candidates.sort((a, b) => b.modifiedMs - a.modifiedMs);
  return candidates[0] || null;
}

async function discoverManifestOptions() {
  const options = [{
    label: "Auto: newest human export",
    path: "",
    source: "auto",
    image_count: null,
    modified_at: null
  }];
  if (await pathExists(HUMAN_RESULTS_DIR)) {
    const entries = await fs.readdir(HUMAN_RESULTS_DIR, { withFileTypes: true });
    const exports = [];
    for (const entry of entries.filter((item) => item.isFile() && item.name.endsWith(".json"))) {
      const filePath = path.join(HUMAN_RESULTS_DIR, entry.name);
      try {
        const parsed = await readJson(filePath);
        const images = parsed?.annotation_set?.manifest?.images;
        if (!Array.isArray(images)) continue;
        const stat = await fs.stat(filePath);
        exports.push({
          label: `${entry.name} (${images.length} images)`,
          path: filePath,
          source: "human_exports",
          image_count: images.length,
          modified_at: stat.mtime.toISOString(),
          modifiedMs: stat.mtimeMs
        });
      } catch {
      // Ignore JSON files that are not aggregate human exports.
      }
    }
    exports.sort((a, b) => b.modifiedMs - a.modifiedMs);
    options.push(...exports.map(({ modifiedMs, ...option }) => option));
  }
  if (await pathExists(APP_V2_DATA_DIR)) {
    const entries = await fs.readdir(APP_V2_DATA_DIR, { withFileTypes: true });
    const manifests = [];
    for (const entry of entries.filter((item) => item.isFile() && item.name.endsWith(".json"))) {
      const filePath = path.join(APP_V2_DATA_DIR, entry.name);
      try {
        const parsed = await readJson(filePath);
        if (!Array.isArray(parsed?.images)) continue;
        const stat = await fs.stat(filePath);
        manifests.push({
          label: `App data: ${entry.name} (${parsed.images.length} images)`,
          path: filePath,
          source: "annotation_app_v2_data",
          image_count: parsed.images.length,
          modified_at: stat.mtime.toISOString(),
          modifiedMs: stat.mtimeMs
        });
      } catch {
        // Ignore JSON files that are not standalone manifests.
      }
    }
    manifests.sort((a, b) => a.label.localeCompare(b.label));
    options.push(...manifests.map(({ modifiedMs, ...option }) => option));
  }
  if (await pathExists(DEFAULT_MANIFEST)) {
    try {
      const parsed = await readJson(DEFAULT_MANIFEST);
      options.push({
        label: `Fallback: ${path.basename(DEFAULT_MANIFEST)} (${parsed.images?.length || 0} images)`,
        path: DEFAULT_MANIFEST,
        source: "fallback",
        image_count: parsed.images?.length || 0,
        modified_at: null
      });
    } catch {
      // Keep the selector usable even if the fallback manifest is malformed.
    }
  }
  return options;
}

async function discoverImageDirOptions() {
  const options = [{
    label: "Auto: manifest path, then known folders",
    path: "",
    source: "auto",
    exists: true
  }];
  for (const [label, dir] of IMAGE_DIR_OPTIONS) {
    options.push({
      label,
      path: dir,
      source: "configured",
      exists: await pathExists(dir)
    });
  }
  return options;
}

async function resolveImagePath(item, manifestDir, imageDir = "") {
  const rawPath = item?.path === undefined || item?.path === null ? "" : String(item.path).trim();
  const filename = item?.filename || (rawPath ? path.basename(rawPath) : "");
  const candidates = [];
  const roots = allowedImageRoots(manifestDir, imageDir);
  if (rawPath) {
    const resolvedRawPath = path.isAbsolute(rawPath) ? path.resolve(rawPath) : path.resolve(manifestDir, rawPath);
    if (roots.some((root) => isWithin(root, resolvedRawPath))) candidates.push(resolvedRawPath);
  }
  if (filename) {
    if (imageDir) candidates.push(path.join(imageDir, filename));
    for (const dir of IMAGE_SEARCH_DIRS) candidates.push(path.join(dir, filename));
  }
  for (const candidate of candidates) {
    if (await pathExists(candidate)) return candidate;
  }
  return "";
}

function imageIdFromRecord(record) {
  const filename =
    record?.image?.filename ||
    record?.image_file ||
    record?.filename ||
    record?.image?.path ||
    record?.image_path;
  const fromFilename = filenameStem(filename);
  if (fromFilename) return fromFilename;
  return filenameStem(record?.image?.image_id || record?.image_id);
}

function chooseLlmAnnotation(record) {
  if (isComparableAnnotation(record?.annotation)) return record.annotation;
  return null;
}

function chooseHumanExportAnnotation(record) {
  if (isComparableAnnotation(record?.annotation)) return record.annotation;
  if (isComparableAnnotation(record?.payload)) return record.payload;
  return null;
}

function sourceRecord(source, imageId, annotation, metadata = {}) {
  return {
    source_id: source.source_id,
    label: source.label,
    short_label: source.short_label,
    type: source.type,
    image_id: imageId,
    annotation,
    metadata
  };
}

async function discoverLlmSources() {
  if (!(await pathExists(LLM_OUTPUT_DIR))) return [];
  const entries = await fs.readdir(LLM_OUTPUT_DIR, { withFileTypes: true });
  const sources = [];
  for (const entry of entries.filter((item) => item.isFile() && item.name.endsWith(".jsonl")).sort((a, b) => a.name.localeCompare(b.name))) {
    const filePath = path.join(LLM_OUTPUT_DIR, entry.name);
    const rawRecords = await readJsonl(filePath);
    const byImage = new Map();
    for (const raw of rawRecords) {
      const imageId = imageIdFromRecord(raw);
      const annotation = chooseLlmAnnotation(raw);
      if (!imageId || !annotation) continue;
      byImage.set(imageId, {
        annotation,
        metadata: {
          ok: raw.ok === true,
          model: raw.model || annotation.machine_annotation?.model || "",
          processed_at: raw.processed_at || annotation.machine_annotation?.processed_at || "",
          task: raw.task || "",
          validation_errors: raw.validation_errors || [],
          normalization_actions: raw.normalization_actions || [],
          source_file: filePath
        }
      });
    }
    if (!byImage.size) continue;
    const base = path.basename(entry.name, ".jsonl");
    const source = {
      source_id: `llm:${slug(base)}`,
      label: base,
      short_label: `L${sources.filter((item) => item.type === "llm").length + 1}`,
      type: "llm",
      path: filePath,
      image_count: byImage.size,
      records: byImage
    };
    sources.push(source);
  }
  return sources;
}

async function readHumanExportRecords(filePath) {
  if (filePath.endsWith(".jsonl")) return readJsonl(filePath);
  const parsed = await readJson(filePath);
  if (Array.isArray(parsed)) return parsed;
  if (Array.isArray(parsed.annotations)) {
    return parsed.annotations.map((record) => ({
      ...record,
      export_schema_version: record.export_schema_version || parsed.export_schema_version,
      annotation_set_id: record.annotation_set_id || parsed.annotation_set?.id,
      annotation_set_name: record.annotation_set_name || parsed.annotation_set?.name,
      task_id: record.task_id || parsed.annotation_set?.task_id
    }));
  }
  return [parsed];
}

async function discoverHumanExportSources() {
  if (!(await pathExists(HUMAN_RESULTS_DIR))) return [];
  const entries = await fs.readdir(HUMAN_RESULTS_DIR, { withFileTypes: true });
  const files = entries
    .filter((item) => item.isFile() && (item.name.endsWith(".json") || item.name.endsWith(".jsonl")))
    .sort((a, b) => a.name.localeCompare(b.name));
  const jsonStems = new Set(files.filter((item) => item.name.endsWith(".json")).map((item) => path.basename(item.name, ".json")));
  const selectedFiles = files.filter((item) => !item.name.endsWith(".jsonl") || !jsonStems.has(path.basename(item.name, ".jsonl")));
  const sources = [];
  for (const file of selectedFiles) {
    const filePath = path.join(HUMAN_RESULTS_DIR, file.name);
    let rawRecords = [];
    try {
      rawRecords = await readHumanExportRecords(filePath);
    } catch {
      continue;
    }
    const byAssignment = new Map();
    for (const raw of rawRecords) {
      const annotation = chooseHumanExportAnnotation(raw);
      const imageId = imageIdFromRecord(raw) || imageIdFromRecord(annotation);
      if (!imageId || !annotation) continue;
      const assignmentCode = String(raw.assignment_code || annotation.session?.session_code || annotation.session?.session_id || "export");
      if (!byAssignment.has(assignmentCode)) byAssignment.set(assignmentCode, new Map());
      byAssignment.get(assignmentCode).set(imageId, {
        annotation,
        metadata: {
          annotator_name: assignmentCode,
          session_id: annotation.session?.session_id || assignmentCode,
          assignment_code: assignmentCode,
          assignment_status: raw.assignment_status || "",
          annotation_set_id: raw.annotation_set_id || "",
          annotation_set_name: raw.annotation_set_name || "",
          task_id: raw.task_id || annotation.task?.task_id || "",
          status: raw.annotation_status || annotation.status || "draft",
          revision: raw.revision || null,
          updated_at: raw.server_updated_at || raw.updated_at || annotation.updated_at || annotation.server_saved_at || "",
          source_file: filePath
        }
      });
    }
    for (const [assignmentCode, byImage] of byAssignment) {
      if (!byImage.size) continue;
      const base = path.basename(file.name).replace(/\.(jsonl|json)$/i, "");
      sources.push({
        source_id: `human:${slug(base)}:${slug(assignmentCode)}`,
        label: `${base} / ${assignmentCode}`,
        short_label: `H${sources.length + 1}`,
        type: "human",
        path: filePath,
        image_count: byImage.size,
        records: byImage
      });
    }
  }
  return sources;
}

async function discoverHumanSources() {
  const sources = await discoverHumanExportSources();
  if (!(await pathExists(HUMAN_SESSIONS_DIR))) return sources;
  const entries = await fs.readdir(HUMAN_SESSIONS_DIR, { withFileTypes: true });
  for (const entry of entries.filter((item) => item.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
    const dir = path.join(HUMAN_SESSIONS_DIR, entry.name);
    const sessionPath = path.join(dir, "session.json");
    const session = (await pathExists(sessionPath)) ? await readJson(sessionPath) : {};
    const files = await fs.readdir(dir, { withFileTypes: true });
    const byImage = new Map();
    for (const file of files.filter((item) => item.isFile() && item.name.endsWith(".json") && item.name !== "session.json")) {
      try {
        const annotation = await readJson(path.join(dir, file.name));
        if (!isComparableAnnotation(annotation)) continue;
        const imageId = imageIdFromRecord(annotation) || path.basename(file.name, ".json");
        byImage.set(imageId, {
          annotation,
          metadata: {
            annotator_name: session.annotator_name || annotation.session?.annotator_name || entry.name,
            session_id: session.session_id || entry.name,
            status: annotation.status || "draft",
            updated_at: annotation.updated_at || annotation.server_saved_at || "",
            source_file: path.join(dir, file.name)
          }
        });
      } catch {
        // Keep other annotations in a session available if one file is malformed.
      }
    }
    if (!byImage.size) continue;
    const annotator = session.annotator_name || entry.name;
    const source = {
      source_id: `human:${slug(entry.name)}`,
      label: `${annotator} (${entry.name})`,
      short_label: `H${sources.length + 1}`,
      type: "human",
      path: dir,
      image_count: byImage.size,
      records: byImage
    };
    sources.push(source);
  }
  return sources;
}

async function discoverSources() {
  const [humans, llms] = await Promise.all([discoverHumanSources(), discoverLlmSources()]);
  return [...humans, ...llms];
}

async function readManifest(manifestPath = null, imageDir = "") {
  const exportManifest = manifestPath ? null : await newestHumanResultManifest();
  const resolved = exportManifest ? exportManifest.sourcePath : path.resolve(manifestPath || DEFAULT_MANIFEST);
  const parsed = exportManifest ? null : await readJson(resolved);
  const manifest = exportManifest ? exportManifest.manifest : (parsed.annotation_set?.manifest || parsed);
  const manifestDir = path.dirname(resolved);
  const images = await Promise.all((manifest.images || []).map(async (item, index) => {
    const rawPath = item.path === undefined || item.path === null ? "" : String(item.path);
    const filename = String(item.filename || (rawPath ? path.basename(rawPath) : item.image_id || item.id || `image_${index + 1}`));
    return {
      image_id: filenameStem(filename) || filenameStem(item.image_id || item.id),
      filename,
      path: await resolveImagePath({ ...item, filename }, manifestDir, imageDir),
      page_type: item.page_type || item.metadata?.page_type || "unknown",
      index,
      total: manifest.images.length,
      metadata: item.metadata || {}
    };
  }));
  return {
    task_id: manifest.task_id || "comparison",
    manifest_path: resolved,
    manifest_source: exportManifest ? "human_exports" : "file",
    images
  };
}

function publicSource(source) {
  return {
    source_id: source.source_id,
    label: source.label,
    short_label: source.short_label,
    type: source.type,
    path: source.path,
    image_count: source.image_count
  };
}

async function bootstrap(manifestPath, imageDir = "") {
  const [manifest, sources] = await Promise.all([readManifest(manifestPath || null, imageDir), discoverSources()]);
  ALLOWED_IMAGE_PATHS.clear();
  for (const image of manifest.images) {
    if (image.path) ALLOWED_IMAGE_PATHS.add(path.resolve(image.path));
  }
  const availability = {};
  for (const image of manifest.images) {
    const available = sources.filter((source) => source.records.has(image.image_id));
    availability[image.image_id] = {
      total: available.length,
      human: available.filter((source) => source.type === "human").length,
      llm: available.filter((source) => source.type === "llm").length,
      source_ids: available.map((source) => source.source_id)
    };
  }
  return { manifest, sources: sources.map(publicSource), availability };
}

async function comparisonForImage(imageId) {
  const sources = await discoverSources();
  const records = [];
  for (const source of sources) {
    const item = source.records.get(imageId);
    if (!item) continue;
    records.push(sourceRecord(source, imageId, item.annotation, item.metadata));
  }
  return records;
}

async function serveFile(res, filePath) {
  try {
    const body = await fs.readFile(filePath);
    res.writeHead(200, { "content-type": contentType(filePath), "cache-control": "no-store" });
    res.end(body);
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "EISDIR") return sendText(res, 404, "Not found");
    throw error;
  }
}

async function streamImage(res, imagePath) {
  if (!imagePath) return sendJson(res, 400, { error: "Image path is required." });
  const resolved = path.resolve(String(imagePath));
  if (!ALLOWED_IMAGE_PATHS.has(resolved)) return sendJson(res, 403, { error: "Image is not in the active manifest." });
  return serveFile(res, resolved);
}

async function allowedSelection(requestedManifest, requestedImageDir) {
  const manifests = await discoverManifestOptions();
  const images = await discoverImageDirOptions();
  const manifest = requestedManifest ? path.resolve(requestedManifest) : "";
  const imageDir = requestedImageDir ? path.resolve(requestedImageDir) : "";
  if (manifest && !manifests.some((option) => option.path && path.resolve(option.path) === manifest)) {
    throw Object.assign(new Error("Manifest is not in a configured manifest directory."), { statusCode: 403 });
  }
  if (imageDir && !images.some((option) => option.path && path.resolve(option.path) === imageDir)) {
    throw Object.assign(new Error("Image directory is not configured."), { statusCode: 403 });
  }
  return { manifest, imageDir };
}

async function handleApi(req, res, pathname, parsedUrl) {
  if (req.method === "GET" && pathname === "/api/health") {
    return sendJson(res, 200, { ok: true, app: "annotation-comparison-app" });
  }
  if (req.method === "GET" && pathname === "/api/manifests") {
    return sendJson(res, 200, { ok: true, manifests: await discoverManifestOptions() });
  }
  if (req.method === "GET" && pathname === "/api/image-dirs") {
    return sendJson(res, 200, { ok: true, image_dirs: await discoverImageDirOptions() });
  }
  if (req.method === "GET" && pathname === "/api/bootstrap") {
    const selection = await allowedSelection(
      parsedUrl.searchParams.get("manifest") || "",
      parsedUrl.searchParams.get("image_dir") || ""
    );
    const data = await bootstrap(
      selection.manifest || null,
      selection.imageDir
    );
    return sendJson(res, 200, { ok: true, ...data });
  }
  if (req.method === "GET" && pathname === "/api/comparison") {
    const imageId = parsedUrl.searchParams.get("image_id");
    if (!imageId) return sendJson(res, 400, { error: "image_id is required" });
    return sendJson(res, 200, { ok: true, image_id: imageId, records: await comparisonForImage(imageId) });
  }
  if (req.method === "GET" && pathname === "/api/manual-matches") {
    return sendJson(res, 200, {
      ok: true,
      image_id: parsedUrl.searchParams.get("image_id") || "",
      matches: await readManualMatches(parsedUrl.searchParams.get("image_id") || "")
    });
  }
  if (req.method === "POST" && pathname === "/api/manual-matches") {
    const body = await readRequestJson(req);
    return sendJson(res, 200, { ok: true, matches: await saveManualMatch(body) });
  }
  if (req.method === "DELETE" && pathname === "/api/manual-matches") {
    const body = await readRequestJson(req);
    return sendJson(res, 200, { ok: true, matches: await deleteManualMatch(body) });
  }
  if (req.method === "GET" && pathname === "/api/image") {
    return streamImage(res, parsedUrl.searchParams.get("path"));
  }
  return sendJson(res, 404, { error: "Unknown API route" });
}

async function handler(req, res) {
  try {
    const parsedUrl = new url.URL(req.url, "http://localhost");
    const pathname = decodeURIComponent(parsedUrl.pathname);
    if (pathname.startsWith("/api/")) return handleApi(req, res, pathname, parsedUrl);
    if (req.method !== "GET") return sendText(res, 405, "Method not allowed");
    if (pathname === "/") return serveFile(res, path.join(PUBLIC_DIR, "index.html"));
    const requested = path.resolve(PUBLIC_DIR, pathname.replace(/^\/+/, ""));
    if (!isWithin(PUBLIC_DIR, requested)) return sendText(res, 403, "Forbidden");
    return serveFile(res, requested);
  } catch (error) {
    console.error(error);
    const status = Number(error.statusCode) || 500;
    return sendJson(res, status, { error: error.message || "Server error" });
  }
}

function listen(port) {
  const server = http.createServer(handler);
  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") return listen(port + 1);
    throw error;
  });
  server.listen(port, HOST, () => console.log(`Annotation Comparison App: http://${HOST}:${port}`));
  return server;
}

if (require.main === module) listen(DEFAULT_PORT);

module.exports = {
  bootstrap,
  comparisonForImage,
  discoverImageDirOptions,
  discoverManifestOptions,
  discoverSources,
  readManualMatches,
  resolveImagePath,
  saveManualMatch,
  imageIdFromRecord,
  isWithin,
  isComparableAnnotation,
  listen
};
