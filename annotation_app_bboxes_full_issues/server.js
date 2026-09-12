const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");
const url = require("node:url");

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, "public");
const NODE_MODULES_DIR = path.join(ROOT, "node_modules");
const TEST_ASSETS_DIR = path.join(ROOT, "test_assets");
const DATA_DIR = path.join(ROOT, "data");
const DEFAULT_PORT = Number(process.env.PORT || 5175);

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".pdf": "application/pdf"
  }[ext] || "application/octet-stream";
}

function sendText(res, status, text) {
  res.writeHead(status, {
    "content-type": "text/plain; charset=utf-8",
    "content-length": Buffer.byteLength(text)
  });
  res.end(text);
}

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body)
  });
  res.end(body);
}

async function readBody(req, limitBytes = 2 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limitBytes) {
      throw new Error("Request body is too large.");
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf-8");
}

function sanitizeStem(value) {
  const stem = String(value || "")
    .replace(/\.pdf$/i, "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!stem) {
    throw new Error("A PDF name is required.");
  }
  return stem.slice(0, 120);
}

function autosaveCsvPath(pdfName) {
  return path.join(DATA_DIR, `${sanitizeStem(pdfName)}_annotations.csv`);
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function parseCsvLine(line) {
  const cells = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"' && line[i + 1] === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }
  cells.push(cell);
  return cells;
}

function normalizeRow(row) {
  const normalized = {
    issue_page_identifier: String(row.issue_page_identifier || ""),
    x1_rel: Number(row.x1_rel),
    y1_rel: Number(row.y1_rel),
    x2_rel: Number(row.x2_rel),
    y2_rel: Number(row.y2_rel),
    analyzable: row.analyzable === true || row.analyzable === "true"
  };
  if (!normalized.issue_page_identifier) {
    throw new Error("Each row requires issue_page_identifier.");
  }
  for (const key of ["x1_rel", "y1_rel", "x2_rel", "y2_rel"]) {
    if (!Number.isFinite(normalized[key])) {
      throw new Error(`Each row requires numeric ${key}.`);
    }
  }
  return normalized;
}

async function readAutosaveCsv(pdfName) {
  const filePath = autosaveCsvPath(pdfName);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const lines = raw.split(/\r?\n/).filter(Boolean);
    if (lines.length <= 1) {
      return { filePath, rows: [] };
    }
    const headers = parseCsvLine(lines[0]);
    const rows = lines.slice(1).map((line) => {
      const values = parseCsvLine(line);
      const row = {};
      for (let i = 0; i < headers.length; i += 1) {
        row[headers[i]] = values[i] ?? "";
      }
      return normalizeRow(row);
    });
    return { filePath, rows };
  } catch (error) {
    if (error.code === "ENOENT") {
      return { filePath, rows: [] };
    }
    throw error;
  }
}

async function writeAutosaveCsv(pdfName, rows) {
  await fs.mkdir(DATA_DIR, { recursive: true });
  const filePath = autosaveCsvPath(pdfName);
  const header = ["issue_page_identifier", "x1_rel", "y1_rel", "x2_rel", "y2_rel", "analyzable"];
  const body = [
    header.join(","),
    ...rows.map((row) => header.map((key) => csvEscape(row[key])).join(","))
  ].join("\n");
  await fs.writeFile(filePath, `${body}\n`, "utf-8");
  return filePath;
}

async function handleApi(req, res, pathname, parsedUrl) {
  if (req.method === "GET" && pathname === "/api/autosave") {
    const pdfName = parsedUrl.searchParams.get("pdf_name");
    const data = await readAutosaveCsv(pdfName);
    return sendJson(res, 200, {
      ok: true,
      rows: data.rows,
      saved_to: data.filePath
    });
  }

  if (req.method === "POST" && pathname === "/api/autosave") {
    const body = JSON.parse(await readBody(req));
    const pdfName = body.pdf_name;
    const rows = Array.isArray(body.rows) ? body.rows.map(normalizeRow) : [];
    const filePath = await writeAutosaveCsv(pdfName, rows);
    return sendJson(res, 200, {
      ok: true,
      row_count: rows.length,
      saved_to: filePath
    });
  }

  return sendJson(res, 404, { error: "Unknown API route." });
}

async function serveFile(res, filePath) {
  try {
    const body = await fs.readFile(filePath);
    res.writeHead(200, { "content-type": contentType(filePath) });
    res.end(body);
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "EISDIR") {
      sendText(res, 404, "Not found");
      return;
    }
    throw error;
  }
}

async function handler(req, res) {
  try {
    const parsedUrl = new url.URL(req.url, `http://${req.headers.host}`);
    const pathname = decodeURIComponent(parsedUrl.pathname);

    if (pathname.startsWith("/api/")) {
      await handleApi(req, res, pathname, parsedUrl);
      return;
    }

    if (req.method !== "GET") {
      sendText(res, 405, "Method not allowed");
      return;
    }

    if (pathname === "/") {
      await serveFile(res, path.join(PUBLIC_DIR, "index.html"));
      return;
    }

    if (pathname.startsWith("/vendor/")) {
      const requested = path.normalize(path.join(NODE_MODULES_DIR, pathname.slice("/vendor/".length)));
      if (!requested.startsWith(NODE_MODULES_DIR)) {
        sendText(res, 403, "Forbidden");
        return;
      }
      await serveFile(res, requested);
      return;
    }

    if (pathname.startsWith("/test_assets/")) {
      const requested = path.normalize(path.join(TEST_ASSETS_DIR, pathname.slice("/test_assets/".length)));
      if (!requested.startsWith(TEST_ASSETS_DIR)) {
        sendText(res, 403, "Forbidden");
        return;
      }
      await serveFile(res, requested);
      return;
    }

    const requested = path.normalize(path.join(PUBLIC_DIR, pathname));
    if (!requested.startsWith(PUBLIC_DIR)) {
      sendText(res, 403, "Forbidden");
      return;
    }
    await serveFile(res, requested);
  } catch (error) {
    console.error(error);
    sendText(res, 500, error.message || "Server error");
  }
}

function startServer(preferredPort = DEFAULT_PORT, attempts = 5) {
  const server = http.createServer(handler);

  server.on("error", (error) => {
    if (error.code === "EADDRINUSE" && attempts > 1) {
      startServer(preferredPort + 1, attempts - 1);
      return;
    }
    throw error;
  });

  server.listen(preferredPort, () => {
    console.log(`Annotation app running at http://localhost:${preferredPort}`);
  });
}

startServer();
