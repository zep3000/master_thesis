import * as pdfjsLib from "/vendor/pdfjs-dist/build/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "/vendor/pdfjs-dist/build/pdf.worker.mjs";

const state = {
  file: null,
  pdfDoc: null,
  currentPage: 1,
  pageCount: 0,
  scale: 1.35,
  ctrlPressed: false,
  annotationsByPage: new Map(),
  drawing: null,
  loadedCsvName: "",
  loadedCsvRowCount: 0
};

const $ = (selector) => document.querySelector(selector);
const canvas = $("#pdfCanvas");
const overlay = $("#overlay");
const canvasWrap = $("#canvasWrap");
const context = canvas.getContext("2d");

function issuePageNumber(pdfPage) {
  return pdfPage - 1;
}

function extractIssueCode(fileName) {
  const stem = String(fileName || "").replace(/\.pdf$/i, "");
  const exact = stem.match(/(19|20)\d{2}[-_.]?([01]\d)[-_.]?([0-3]\d)/);
  if (exact) {
    return `${exact[0].slice(0, 4)}-${exact[2]}${exact[3]}`.replace(/[-_.]?([01]\d)[-_.]?([0-3]\d)$/, `-${exact[2]}${exact[3]}`);
  }
  const compact = stem.match(/((19|20)\d{2})([01]\d)([0-3]\d)/);
  if (compact) {
    return `${compact[1]}-${compact[3]}${compact[4]}`;
  }
  return stem;
}

function issueIdentifier(pdfPage) {
  const issueCode = extractIssueCode(state.file?.name || "issue");
  const pageCode = String(issuePageNumber(pdfPage)).padStart(4, "0");
  return `${issueCode}-${pageCode}`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function roundRelative(value) {
  return Number(value.toFixed(6));
}

function getPageAnnotations(pageNumber = state.currentPage) {
  if (!state.annotationsByPage.has(pageNumber)) {
    state.annotationsByPage.set(pageNumber, []);
  }
  return state.annotationsByPage.get(pageNumber);
}

function allAnnotations() {
  return [...state.annotationsByPage.entries()]
    .sort((a, b) => a[0] - b[0])
    .flatMap(([, rows]) => rows);
}

function setAutosaveStatus(status, detail = "") {
  const element = $("#autosaveStatus");
  if (!element) {
    return;
  }
  const text = {
    idle: "Autosave idle",
    saving: "Autosaving...",
    loaded: detail ? `Loaded CSV: ${detail}` : "Loaded existing CSV",
    saved: detail ? `Autosaved: ${detail}` : "Autosaved",
    error: detail ? `Autosave failed: ${detail}` : "Autosave failed"
  }[status] || "Autosave idle";
  const tone = status === "error"
    ? "danger"
    : status === "saving"
      ? "neutral"
      : status === "saved" || status === "loaded"
        ? "saved"
        : "";
  element.textContent = text;
  element.className = `status-pill ${tone}`.trim();
}

function updateModifierStatus() {
  const element = $("#modifierStatus");
  const isNo = state.ctrlPressed || state.drawing?.modifierUsed;
  element.textContent = isNo ? "BBox mode: analyzable no" : "BBox mode: analyzable yes";
  element.className = `status-pill ${isNo ? "danger" : "neutral"}`;
}

function updatePageStatus() {
  const status = $("#pageStatus");
  if (!state.pdfDoc) {
    status.textContent = "No PDF loaded";
    return;
  }
  status.textContent = `PDF page ${state.currentPage} / ${state.pageCount} | Issue page ${issuePageNumber(state.currentPage)}`;
}

function updatePageSummary() {
  const summary = $("#pageSummary");
  if (!state.pdfDoc) {
    summary.textContent = "No page loaded.";
    return;
  }
  const count = getPageAnnotations().length;
  summary.innerHTML = `
    <strong>${state.file.name}</strong><br>
    PDF page: ${state.currentPage}<br>
    Issue page: ${issuePageNumber(state.currentPage)}<br>
    Stored as: ${issueIdentifier(state.currentPage)}<br>
    CSV loaded: ${state.loadedCsvName ? `${state.loadedCsvName} (${state.loadedCsvRowCount} rows)` : "no existing CSV"}<br>
    Saved boxes on this page: ${count}
  `;
}

function setControlsDisabled(disabled) {
  $("#prevPageButton").disabled = disabled;
  $("#nextPageButton").disabled = disabled;
  $("#clearPageButton").disabled = disabled;
  $("#exportCsvButton").disabled = disabled;
}

function renderAnnotationList() {
  const rows = getPageAnnotations();
  $("#boxCount").textContent = String(rows.length);
  const list = $("#annotationList");
  if (!rows.length) {
    list.className = "annotation-list empty";
    list.textContent = "No boxes on this page.";
    return;
  }

  list.className = "annotation-list";
  list.innerHTML = rows.map((row, index) => `
    <div class="annotation-row ${row.analyzable ? "" : "bad"}">
      <p class="annotation-meta">
        #${index + 1} | ${row.issue_page_identifier} | ${row.analyzable ? "analyzable yes" : "analyzable no"}
      </p>
      <p class="annotation-coords">
        x1=${row.x1_rel.toFixed(4)} y1=${row.y1_rel.toFixed(4)} x2=${row.x2_rel.toFixed(4)} y2=${row.y2_rel.toFixed(4)}
      </p>
      <div class="annotation-actions">
        <button type="button" data-delete-index="${index}">Delete</button>
      </div>
    </div>
  `).join("");
}

function svgRect(rect, className) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  element.setAttribute("x", String(rect.x));
  element.setAttribute("y", String(rect.y));
  element.setAttribute("width", String(rect.width));
  element.setAttribute("height", String(rect.height));
  element.setAttribute("class", className);
  return element;
}

function renderOverlay() {
  overlay.innerHTML = "";
  overlay.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);

  for (const row of getPageAnnotations()) {
    const rect = {
      x: row.x1_rel * canvas.width,
      y: row.y1_rel * canvas.height,
      width: (row.x2_rel - row.x1_rel) * canvas.width,
      height: (row.y2_rel - row.y1_rel) * canvas.height
    };
    overlay.appendChild(svgRect(rect, `bbox ${row.analyzable ? "" : "not-analyzable"}`.trim()));
  }

  if (state.drawing?.rect) {
    overlay.appendChild(svgRect(state.drawing.rect, `bbox pending ${state.drawing.modifierUsed ? "not-analyzable" : ""}`.trim()));
  }
}

async function renderPage() {
  if (!state.pdfDoc) {
    return;
  }
  const page = await state.pdfDoc.getPage(state.currentPage);
  const viewport = page.getViewport({ scale: state.scale });
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  canvasWrap.style.width = `${canvas.width}px`;
  canvasWrap.style.height = `${canvas.height}px`;
  await page.render({ canvasContext: context, viewport }).promise;
  renderOverlay();
  updatePageStatus();
  updatePageSummary();
  renderAnnotationList();
  $("#prevPageButton").disabled = state.currentPage <= 1;
  $("#nextPageButton").disabled = state.currentPage >= state.pageCount;
}

function groupRowsByPdfPage(rows) {
  const grouped = new Map();
  for (const row of rows) {
    const match = String(row.issue_page_identifier || "").match(/-(\d{4})$/);
    if (!match) {
      continue;
    }
    const issuePage = Number(match[1]);
    const pdfPage = issuePage + 1;
    if (!grouped.has(pdfPage)) {
      grouped.set(pdfPage, []);
    }
    grouped.get(pdfPage).push({
      issue_page_identifier: row.issue_page_identifier,
      x1_rel: Number(row.x1_rel),
      y1_rel: Number(row.y1_rel),
      x2_rel: Number(row.x2_rel),
      y2_rel: Number(row.y2_rel),
      analyzable: row.analyzable === true || row.analyzable === "true"
    });
  }
  return grouped;
}

async function saveAutosave() {
  if (!state.file) {
    return;
  }
  setAutosaveStatus("saving");
  try {
    const response = await fetch("/api/autosave", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        pdf_name: state.file.name,
        rows: allAnnotations()
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed");
    }
    const fileName = data.saved_to.split(/[\\/]/).pop();
    state.loadedCsvName = fileName;
    state.loadedCsvRowCount = allAnnotations().length;
    setAutosaveStatus("saved", fileName);
    updatePageSummary();
  } catch (error) {
    console.error(error);
    setAutosaveStatus("error", error.message);
  }
}

async function loadAutosave() {
  if (!state.file) {
    return;
  }
  const response = await fetch(`/api/autosave?pdf_name=${encodeURIComponent(state.file.name)}`);
  const data = await response.json();
  state.annotationsByPage = groupRowsByPdfPage(Array.isArray(data.rows) ? data.rows : []);
  state.loadedCsvName = data.saved_to.split(/[\\/]/).pop();
  state.loadedCsvRowCount = Array.isArray(data.rows) ? data.rows.length : 0;
  setAutosaveStatus(
    state.loadedCsvRowCount > 0 ? "loaded" : "idle",
    state.loadedCsvRowCount > 0 ? `${state.loadedCsvName} (${state.loadedCsvRowCount} rows)` : ""
  );
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const x = clamp((event.clientX - rect.left) * (canvas.width / rect.width), 0, canvas.width);
  const y = clamp((event.clientY - rect.top) * (canvas.height / rect.height), 0, canvas.height);
  return { x, y };
}

function updatePendingRect(event) {
  if (!state.drawing) {
    return;
  }
  const current = pointerPosition(event);
  state.drawing.modifierUsed = state.drawing.modifierUsed || state.ctrlPressed || event.ctrlKey;
  const left = Math.min(state.drawing.start.x, current.x);
  const top = Math.min(state.drawing.start.y, current.y);
  const width = Math.abs(current.x - state.drawing.start.x);
  const height = Math.abs(current.y - state.drawing.start.y);
  state.drawing.rect = { x: left, y: top, width, height };
  updateModifierStatus();
  renderOverlay();
}

function abortCurrentDrawing() {
  if (!state.drawing) {
    return;
  }
  state.drawing = null;
  updateModifierStatus();
  renderOverlay();
}

function commitCurrentDrawing() {
  const drawing = state.drawing;
  if (!drawing || !drawing.rect) {
    abortCurrentDrawing();
    return;
  }
  if (drawing.rect.width < 6 || drawing.rect.height < 6) {
    abortCurrentDrawing();
    return;
  }

  const rows = getPageAnnotations();
  rows.push({
    issue_page_identifier: issueIdentifier(state.currentPage),
    x1_rel: roundRelative(drawing.rect.x / canvas.width),
    y1_rel: roundRelative(drawing.rect.y / canvas.height),
    x2_rel: roundRelative((drawing.rect.x + drawing.rect.width) / canvas.width),
    y2_rel: roundRelative((drawing.rect.y + drawing.rect.height) / canvas.height),
    analyzable: !drawing.modifierUsed
  });
  state.drawing = null;
  updateModifierStatus();
  renderOverlay();
  updatePageSummary();
  renderAnnotationList();
  void saveAutosave();
}

async function goToPage(pageNumber) {
  if (!state.pdfDoc) {
    return;
  }
  const next = clamp(pageNumber, 1, state.pageCount);
  if (next === state.currentPage) {
    return;
  }
  state.currentPage = next;
  abortCurrentDrawing();
  await renderPage();
}

function csvEscape(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportCsv() {
  const rows = allAnnotations();
  const header = ["issue_page_identifier", "x1_rel", "y1_rel", "x2_rel", "y2_rel", "analyzable"];
  const lines = [
    header.join(","),
    ...rows.map((row) => header.map((key) => csvEscape(row[key])).join(","))
  ];
  const blob = new Blob([`${lines.join("\n")}\n`], { type: "text/csv;charset=utf-8" });
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const base = state.file?.name?.replace(/\.pdf$/i, "") || "annotations";
  link.href = href;
  link.download = `${base}_annotations.csv`;
  link.click();
  URL.revokeObjectURL(href);
}

async function loadPdfData(file, data) {
  state.file = file;
  state.annotationsByPage = new Map();
  state.currentPage = 1;
  abortCurrentDrawing();
  const loadingTask = pdfjsLib.getDocument({ data });
  state.pdfDoc = await loadingTask.promise;
  state.pageCount = state.pdfDoc.numPages;
  setControlsDisabled(false);
  await loadAutosave();
  await renderPage();
}

async function loadPdfFile(file) {
  const data = new Uint8Array(await file.arrayBuffer());
  await loadPdfData(file, data);
}

async function maybeLoadPdfFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const pdfUrl = params.get("pdf");
  if (!pdfUrl) {
    return;
  }
  const response = await fetch(pdfUrl);
  if (!response.ok) {
    throw new Error(`Failed to load PDF from ${pdfUrl}`);
  }
  const blob = await response.blob();
  const fileName = pdfUrl.split("/").pop() || "issue.pdf";
  const file = new File([blob], fileName, { type: "application/pdf" });
  await loadPdfData(file, new Uint8Array(await blob.arrayBuffer()));
}

function bindEvents() {
  $("#pdfInput").addEventListener("change", async (event) => {
    const [file] = event.target.files || [];
    if (!file) {
      return;
    }
    await loadPdfFile(file);
  });

  $("#prevPageButton").addEventListener("click", () => goToPage(state.currentPage - 1));
  $("#nextPageButton").addEventListener("click", () => goToPage(state.currentPage + 1));
  $("#clearPageButton").addEventListener("click", () => {
    state.annotationsByPage.set(state.currentPage, []);
    renderOverlay();
    updatePageSummary();
    renderAnnotationList();
    void saveAutosave();
  });
  $("#exportCsvButton").addEventListener("click", exportCsv);

  overlay.addEventListener("pointerdown", (event) => {
    if (!state.pdfDoc || event.button !== 0) {
      return;
    }
    event.preventDefault();
    overlay.setPointerCapture(event.pointerId);
    state.drawing = {
      start: pointerPosition(event),
      rect: null,
      modifierUsed: state.ctrlPressed || event.ctrlKey
    };
    updatePendingRect(event);
  });

  overlay.addEventListener("pointermove", (event) => {
    if (!state.drawing) {
      return;
    }
    event.preventDefault();
    updatePendingRect(event);
  });

  overlay.addEventListener("pointerup", (event) => {
    if (!state.drawing || event.button !== 0) {
      return;
    }
    event.preventDefault();
    if (overlay.hasPointerCapture(event.pointerId)) {
      overlay.releasePointerCapture(event.pointerId);
    }
    updatePendingRect(event);
    commitCurrentDrawing();
  });

  overlay.addEventListener("pointercancel", abortCurrentDrawing);
  overlay.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    abortCurrentDrawing();
  });

  window.addEventListener("keydown", async (event) => {
    if (event.key === "Control") {
      state.ctrlPressed = true;
      updateModifierStatus();
      renderOverlay();
      return;
    }
    if (!state.pdfDoc) {
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      await goToPage(state.currentPage - 1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      await goToPage(state.currentPage + 1);
    }
  });

  window.addEventListener("keyup", (event) => {
    if (event.key === "Control") {
      state.ctrlPressed = false;
      updateModifierStatus();
      renderOverlay();
    }
  });

  $("#annotationList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-index]");
    if (!button) {
      return;
    }
    const index = Number(button.dataset.deleteIndex);
    const rows = getPageAnnotations();
    rows.splice(index, 1);
    renderOverlay();
    updatePageSummary();
    renderAnnotationList();
    void saveAutosave();
  });
}

function boot() {
  setControlsDisabled(true);
  updateModifierStatus();
  updatePageStatus();
  updatePageSummary();
  renderAnnotationList();
  bindEvents();
}

boot();
maybeLoadPdfFromQuery().catch((error) => {
  console.error(error);
  $("#pageSummary").textContent = error.message;
});
