const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  bootstrap: null,
  session: null,
  items: [],
  statuses: {},
  wrongIdentifications: {},
  index: 0,
  annotation: null,
  image: null,
  contextView: true,
  showFaceBoxes: true,
  zoom: 1,
  saveTimer: null,
  saveInFlight: null,
  saving: false,
  dirty: false,
  changeVersion: 0
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { "content-type": "application/json" } : undefined,
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function setStatus(selector, text, kind = "") {
  const element = $(selector);
  element.textContent = text;
  element.className = `status ${kind}`.trim();
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function defaultAnnotation() {
  return {
    identification_wrong: false,
    identification_note: "",
    no_faces: false,
    category_review: null,
    corrected_brand_category: null,
    brand_name_review: null,
    corrected_brand_name: "",
    notes: ""
  };
}

function currentItem() { return state.items[state.index]; }

function completedCount() {
  return Object.values(state.statuses).filter((status) => status === "complete").length;
}

function updateProgress() {
  const complete = completedCount();
  const total = state.items.length;
  $("#sidebarProgress").textContent = `${complete} / ${total} complete`;
  $("#footerProgress").textContent = `${complete} / ${total}`;
  $("#progressBar").value = total ? (complete / total) * 100 : 0;
}

function renderItemList() {
  const query = $("#itemSearch").value.trim().toLowerCase();
  const fragment = document.createDocumentFragment();
  state.items.forEach((item, index) => {
    const haystack = `${item.image_id} ${item.filename} ${item.advertisement_id} ${item.assigned_brand_name}`.toLowerCase();
    if (query && !haystack.includes(query)) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `item-button${index === state.index ? " active" : ""}`;
    button.dataset.index = index;
    const status = state.statuses[item.item_id] || "not_started";
    button.innerHTML = `
      <i class="dot ${status === "not_started" ? "not-started" : status}"></i>
      <span class="item-copy"><strong>${escapeHtml(item.image_id)}</strong><small>${escapeHtml(item.advertisement_id)} · ${escapeHtml(item.assigned_brand_name)}</small></span>
      <span class="item-number">${index + 1}</span>`;
    fragment.append(button);
  });
  $("#itemList").replaceChildren(fragment);
  updateProgress();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function syncAnnotationFromForm() {
  if (!state.annotation) return;
  state.annotation.identification_wrong = $("#identificationWrong").checked;
  state.annotation.identification_note = $("#identificationNote").value;
  state.annotation.no_faces = $("#noFaces").checked;
  state.annotation.corrected_brand_category = $("#categoryCorrection").value || null;
  state.annotation.corrected_brand_name = $("#brandCorrection").value;
  state.annotation.notes = $("#notes").value;
}

function syncForm() {
  const a = state.annotation;
  $("#identificationWrong").checked = Boolean(a.identification_wrong);
  $("#identificationNote").value = a.identification_note || "";
  $("#noFaces").checked = Boolean(a.no_faces);
  $("#identificationNoteField").classList.toggle("hidden", !a.identification_wrong);
  $("#fieldReviews").classList.toggle("disabled", Boolean(a.identification_wrong));
  $("#categoryCorrection").value = a.corrected_brand_category || "";
  $("#brandCorrection").value = a.corrected_brand_name || "";
  $("#notes").value = a.notes || "";
  $("#categoryCorrectionField").classList.toggle("hidden", a.category_review !== "incorrect");
  $("#brandCorrectionField").classList.toggle("hidden", a.brand_name_review !== "incorrect");
  $$(".verdict-control button").forEach((button) => {
    const value = button.dataset.field === "category" ? a.category_review : a.brand_name_review;
    button.classList.toggle("selected", button.dataset.verdict === value);
    button.setAttribute("aria-pressed", String(button.dataset.verdict === value));
  });
}

async function loadItem(index, { saveFirst = true } = {}) {
  if (index < 0 || index >= state.items.length) return;
  if (saveFirst && state.dirty) await saveDraft();
  state.index = index;
  const item = currentItem();
  setStatus("#saveStatus", "Loading…");
  const data = await api(`/api/annotation?session_id=${encodeURIComponent(state.session.session_id)}&item_id=${encodeURIComponent(item.item_id)}`);
  state.annotation = data.annotation ? { ...defaultAnnotation(), ...data.annotation } : defaultAnnotation();
  state.dirty = false;
  state.changeVersion += 1;
  state.contextView = true;
  state.zoom = 1;
  $("#itemPosition").textContent = `Ad ${index + 1} of ${state.items.length}`;
  $("#imageTitle").textContent = item.filename;
  const year = item.image_metadata?.year;
  $("#imageMetadata").textContent = year ? `The Economist · ${year}` : "The Economist";
  $("#assignedCategory").textContent = item.assigned_brand_category;
  $("#assignedBrand").textContent = item.assigned_brand_name;
  $("#previousButton").disabled = index === 0;
  $("#saveNextButton").textContent = index === state.items.length - 1 ? "Save & finish" : "Save & next";
  syncForm();
  renderItemList();
  await loadImage(item);
  setStatus("#saveStatus", data.annotation?.status === "complete" ? "Saved as complete." : "");
  const active = $(".item-button.active");
  active?.scrollIntoView({ block: "nearest" });
}

async function loadImage(item) {
  $("#imageLoading").classList.remove("hidden");
  const image = new Image();
  image.decoding = "async";
  image.src = `/api/image?image_id=${encodeURIComponent(item.image_id)}`;
  await image.decode().catch(() => new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; }));
  if (currentItem()?.item_id !== item.item_id) return;
  state.image = image;
  drawImage();
  $("#imageLoading").classList.add("hidden");
}

function drawImage() {
  if (!state.image || !currentItem()) return;
  const canvas = $("#adCanvas");
  const context = canvas.getContext("2d");
  const image = state.image;
  const [x1, y1, x2, y2] = currentItem().bbox_1000.map((value) => Number(value) / 1000);
  let sx = 0, sy = 0, sw = image.naturalWidth, sh = image.naturalHeight;
  if (!state.contextView) {
    const pad = 0.025;
    const left = Math.max(0, x1 - pad);
    const top = Math.max(0, y1 - pad);
    const right = Math.min(1, x2 + pad);
    const bottom = Math.min(1, y2 + pad);
    sx = Math.round(left * image.naturalWidth);
    sy = Math.round(top * image.naturalHeight);
    sw = Math.max(1, Math.round((right - left) * image.naturalWidth));
    sh = Math.max(1, Math.round((bottom - top) * image.naturalHeight));
  }
  const maxDimension = 1800;
  const scale = Math.min(1, maxDimension / Math.max(sw, sh));
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  context.imageSmoothingQuality = "high";
  context.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  if (state.contextView) {
    context.save();
    context.strokeStyle = "#e34b3f";
    context.lineWidth = Math.max(3, canvas.width / 280);
    context.setLineDash([Math.max(8, canvas.width / 90), Math.max(5, canvas.width / 150)]);
    context.strokeRect(x1 * canvas.width, y1 * canvas.height, (x2 - x1) * canvas.width, (y2 - y1) * canvas.height);
    context.restore();
  }
  if (state.showFaceBoxes) {
    const faceBoxes = currentItem().face_boxes_1000 || [];
    faceBoxes.forEach((face, index) => {
      const [fx1, fy1, fx2, fy2] = face.bbox_1000.map((value) => Number(value) / 1000);
      const sourceX1 = fx1 * image.naturalWidth;
      const sourceY1 = fy1 * image.naturalHeight;
      const sourceX2 = fx2 * image.naturalWidth;
      const sourceY2 = fy2 * image.naturalHeight;
      if (sourceX2 <= sx || sourceY2 <= sy || sourceX1 >= sx + sw || sourceY1 >= sy + sh) return;
      const cx1 = ((sourceX1 - sx) / sw) * canvas.width;
      const cy1 = ((sourceY1 - sy) / sh) * canvas.height;
      const cx2 = ((sourceX2 - sx) / sw) * canvas.width;
      const cy2 = ((sourceY2 - sy) / sh) * canvas.height;
      context.save();
      context.strokeStyle = "#1683c4";
      context.fillStyle = "rgba(22, 131, 196, 0.11)";
      context.lineWidth = Math.max(3, canvas.width / 300);
      context.setLineDash([]);
      context.fillRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
      context.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
      const label = `Face ${index + 1}`;
      const fontSize = Math.max(13, Math.min(25, canvas.width / 55));
      context.font = `700 ${fontSize}px Arial`;
      const labelWidth = context.measureText(label).width + 12;
      const labelY = Math.max(0, cy1 - fontSize - 7);
      context.fillStyle = "#1683c4";
      context.fillRect(cx1, labelY, labelWidth, fontSize + 7);
      context.fillStyle = "#fff";
      context.fillText(label, cx1 + 6, labelY + fontSize);
      context.restore();
    });
  }
  const faceCount = (currentItem().face_boxes_1000 || []).length;
  $("#faceBoxesToggle").disabled = faceCount === 0;
  $("#faceBoxesToggle").textContent = faceCount === 0
    ? "No model face boxes"
    : `${state.showFaceBoxes ? "Hide" : "Show"} face ${faceCount === 1 ? "box" : "boxes"} (${faceCount})`;
  $("#faceBoxesToggle").setAttribute("aria-pressed", String(faceCount > 0 && state.showFaceBoxes));
  applyZoom();
  $("#viewToggle").textContent = state.contextView ? "Zoom to model box" : "Show full page";
  $("#viewToggle").title = state.contextView
    ? "Show a close crop of the model-provided box; it may not contain the whole advertisement."
    : "Show the complete source page with the model-provided box highlighted.";
  $("#adId").textContent = state.contextView
    ? `${currentItem().advertisement_id} · full page · model box highlighted`
    : `${currentItem().advertisement_id} · model-box crop · surrounding ad content may be omitted`;
}

function applyZoom() {
  const canvas = $("#adCanvas");
  const scroll = $("#imageScroll");
  const available = Math.max(260, scroll.clientWidth - 48);
  const fitWidth = Math.min(canvas.width, available);
  canvas.style.width = `${Math.round(fitWidth * state.zoom)}px`;
  canvas.style.height = "auto";
}

function zoomImageAtPointer(deltaY, clientX, clientY) {
  const canvas = $("#adCanvas");
  const scroll = $("#imageScroll");
  if (!canvas.width || !canvas.height) return;
  const before = canvas.getBoundingClientRect();
  const anchorX = Math.max(0, Math.min(1, (clientX - before.left) / Math.max(1, before.width)));
  const anchorY = Math.max(0, Math.min(1, (clientY - before.top) / Math.max(1, before.height)));
  const factor = Math.max(0.8, Math.min(1.25, Math.exp(-deltaY * 0.0025)));
  const nextZoom = Math.max(0.4, Math.min(3, state.zoom * factor));
  if (Math.abs(nextZoom - state.zoom) < 0.001) return;
  state.zoom = nextZoom;
  applyZoom();
  const after = canvas.getBoundingClientRect();
  scroll.scrollLeft += after.left + anchorX * after.width - clientX;
  scroll.scrollTop += after.top + anchorY * after.height - clientY;
}

function markDirty() {
  syncAnnotationFromForm();
  state.dirty = true;
  state.changeVersion += 1;
  if ((state.statuses[currentItem().item_id] || "not_started") === "not_started") {
    state.statuses[currentItem().item_id] = "started";
    renderItemList();
  }
  setStatus("#saveStatus", "Unsaved changes…");
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveDraft().catch(showSaveError), 650);
}

function currentAnnotationCanComplete() {
  const annotation = state.annotation;
  if (!annotation) return false;
  if (annotation.identification_wrong) return true;
  if (!["correct", "incorrect"].includes(annotation.category_review)) return false;
  if (annotation.category_review === "incorrect" && !annotation.corrected_brand_category) return false;
  if (!["correct", "incorrect"].includes(annotation.brand_name_review)) return false;
  if (annotation.brand_name_review === "incorrect" && !String(annotation.corrected_brand_name || "").trim()) return false;
  return true;
}

async function saveDraft() {
  if (!state.annotation || !state.dirty) return;
  clearTimeout(state.saveTimer);
  if (state.saveInFlight) {
    try { await state.saveInFlight; } catch {}
    if (state.dirty) return saveDraft();
    return;
  }
  syncAnnotationFromForm();
  const item = currentItem();
  const snapshot = JSON.parse(JSON.stringify(state.annotation));
  const savedVersion = state.changeVersion;
  state.saving = true;
  setStatus("#saveStatus", "Saving…");
  const request = api("/api/annotation", {
    method: "POST",
    body: { session_id: state.session.session_id, item_id: item.item_id, annotation: snapshot, complete: false }
  });
  state.saveInFlight = request;
  let savedSuccessfully = false;
  try {
    const result = await request;
    savedSuccessfully = true;
    if (currentItem().item_id === item.item_id) {
      state.statuses[item.item_id] = result.annotation.status === "complete" ? "complete" : "started";
      state.wrongIdentifications[item.item_id] = Boolean(result.annotation.identification_wrong);
      if (state.changeVersion === savedVersion) {
        state.annotation = { ...state.annotation, ...result.annotation };
        state.dirty = false;
        setStatus("#saveStatus", "Saved.", "success");
      } else {
        setStatus("#saveStatus", "Unsaved changes…");
      }
      renderItemList();
    }
  } finally {
    if (state.saveInFlight === request) state.saveInFlight = null;
    state.saving = false;
    if (savedSuccessfully && currentItem()?.item_id === item.item_id && state.dirty) {
      clearTimeout(state.saveTimer);
      state.saveTimer = setTimeout(() => saveDraft().catch(showSaveError), 50);
    }
  }
}

function showSaveError(error) {
  setStatus("#saveStatus", error.message, "error");
}

async function saveCompleteAndNext() {
  clearTimeout(state.saveTimer);
  if (state.saveInFlight) {
    try { await state.saveInFlight; } catch {}
  }
  clearTimeout(state.saveTimer);
  syncAnnotationFromForm();
  const item = currentItem();
  $("#saveNextButton").disabled = true;
  state.saving = true;
  setStatus("#saveStatus", "Saving…");
  const request = api("/api/annotation", {
    method: "POST",
    body: { session_id: state.session.session_id, item_id: item.item_id, annotation: state.annotation, complete: true }
  });
  state.saveInFlight = request;
  try {
    const result = await request;
    state.annotation = result.annotation;
    state.dirty = false;
    state.statuses[item.item_id] = "complete";
    state.wrongIdentifications[item.item_id] = Boolean(result.annotation.identification_wrong);
    renderItemList();
    if (completedCount() === state.items.length) return showCompletion();
    const nextUnfinished = state.items.findIndex((candidate, index) => index > state.index && state.statuses[candidate.item_id] !== "complete");
    const fallback = state.items.findIndex((candidate) => state.statuses[candidate.item_id] !== "complete");
    await loadItem(nextUnfinished >= 0 ? nextUnfinished : fallback, { saveFirst: false });
  } catch (error) {
    setStatus("#saveStatus", error.message, "error");
  } finally {
    if (state.saveInFlight === request) state.saveInFlight = null;
    state.saving = false;
    $("#saveNextButton").disabled = false;
  }
}

function exportResults() {
  window.location.href = `/api/export?session_id=${encodeURIComponent(state.session.session_id)}`;
}

function showCompletion() {
  $("#appView").classList.add("hidden");
  $("#completionView").classList.remove("hidden");
  $("#completedCount").textContent = completedCount();
  $("#completionSession").textContent = state.session.session_id;
  $("#wrongIdCount").textContent = Object.values(state.wrongIdentifications).filter(Boolean).length;
  $("#completionView").focus();
}

function showWorkspace() {
  $("#welcomeView").classList.add("hidden");
  $("#completionView").classList.add("hidden");
  $("#appView").classList.remove("hidden");
}

function showWelcome() {
  clearTimeout(state.saveTimer);
  $("#appView").classList.add("hidden");
  $("#completionView").classList.add("hidden");
  $("#welcomeView").classList.remove("hidden");
  refreshBootstrap().catch((error) => setStatus("#welcomeStatus", error.message, "error"));
}

async function openSession(session, newlyCreated) {
  state.session = session;
  $("#sessionCode").textContent = session.session_id;
  const data = await api(`/api/items?session_id=${encodeURIComponent(session.session_id)}`);
  state.items = data.items;
  state.statuses = data.statuses;
  state.wrongIdentifications = data.wrong_identifications || {};
  const firstIncomplete = state.items.findIndex((item) => state.statuses[item.item_id] !== "complete");
  state.index = firstIncomplete >= 0 ? firstIncomplete : 0;
  showWorkspace();
  await loadItem(state.index, { saveFirst: false });
  if (newlyCreated) {
    $("#dialogSessionCode").textContent = session.session_id;
    $("#sessionDialog").showModal();
  } else if (firstIncomplete < 0) {
    showCompletion();
  }
}

async function refreshBootstrap() {
  state.bootstrap = await api("/api/bootstrap");
  $("#welcomeImages").textContent = state.bootstrap.task.image_count;
  $("#welcomeItems").textContent = state.bootstrap.task.item_count;
  const select = $("#sessionSelect");
  select.innerHTML = state.bootstrap.sessions.length
    ? state.bootstrap.sessions.map((session) => `<option value="${session.session_id}">${session.session_id} — ${session.complete}/${session.total} done — ${escapeHtml(formatDate(session.updated_at))}</option>`).join("")
    : '<option value="">No saved sessions</option>';
  $("#categoryCorrection").innerHTML = '<option value="">Choose a category…</option>' + state.bootstrap.brand_categories.map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join("");
}

async function startSelectedSession() {
  const mode = $(".mode-button.selected").dataset.mode;
  setStatus("#welcomeStatus", mode === "new" ? "Creating session…" : "Opening session…");
  $("#startButton").disabled = true;
  try {
    const body = mode === "resume" ? { session_id: $("#sessionSelect").value } : {};
    if (mode === "resume" && !body.session_id) throw new Error("There is no saved session to resume.");
    const result = await api("/api/session", { method: "POST", body });
    await openSession(result.session, !result.resumed);
    setStatus("#welcomeStatus", "");
  } catch (error) {
    setStatus("#welcomeStatus", error.message, "error");
  } finally {
    $("#startButton").disabled = false;
  }
}

$$('.mode-button').forEach((button) => button.addEventListener("click", () => {
  $$(".mode-button").forEach((candidate) => {
    const selected = candidate === button;
    candidate.classList.toggle("selected", selected);
    candidate.setAttribute("aria-pressed", String(selected));
  });
  const resume = button.dataset.mode === "resume";
  $("#resumeArea").classList.toggle("hidden", !resume);
  $("#startButton").textContent = resume ? "Resume verification" : "Start new verification";
}));

$("#startButton").addEventListener("click", startSelectedSession);
$("#continueSessionButton").addEventListener("click", () => $("#sessionDialog").close());
$("#copySessionButton").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.session.session_id);
  $("#copySessionButton").textContent = "Copied";
});
$("#itemList").addEventListener("click", (event) => {
  const button = event.target.closest(".item-button");
  if (button) loadItem(Number(button.dataset.index)).catch(showSaveError);
});
$("#itemSearch").addEventListener("input", renderItemList);

$$(".verdict-control button").forEach((button) => button.addEventListener("click", async () => {
  if (button.dataset.field === "category") state.annotation.category_review = button.dataset.verdict;
  else state.annotation.brand_name_review = button.dataset.verdict;
  syncForm();
  markDirty();
  if (button.dataset.verdict === "incorrect") {
    (button.dataset.field === "category" ? $("#categoryCorrection") : $("#brandCorrection")).focus();
  } else if (button.dataset.field === "brand" && currentAnnotationCanComplete()) {
    await saveCompleteAndNext();
  }
}));

$("#identificationWrong").addEventListener("change", () => { syncAnnotationFromForm(); syncForm(); markDirty(); });
$("#noFaces").addEventListener("change", markDirty);
["#identificationNote", "#categoryCorrection", "#brandCorrection", "#notes"].forEach((selector) => {
  $(selector).addEventListener(selector.includes("Correction") && selector !== "#brandCorrection" ? "change" : "input", markDirty);
});
$("#saveNextButton").addEventListener("click", saveCompleteAndNext);
$("#previousButton").addEventListener("click", () => loadItem(state.index - 1).catch(showSaveError));
$("#viewToggle").addEventListener("click", () => { state.contextView = !state.contextView; state.zoom = 1; drawImage(); });
$("#faceBoxesToggle").addEventListener("click", () => { state.showFaceBoxes = !state.showFaceBoxes; drawImage(); });
$("#zoomIn").addEventListener("click", () => { state.zoom = Math.min(3, state.zoom + .2); applyZoom(); });
$("#zoomOut").addEventListener("click", () => { state.zoom = Math.max(.4, state.zoom - .2); applyZoom(); });
$("#zoomFit").addEventListener("click", () => { state.zoom = 1; applyZoom(); });
$("#imageScroll").addEventListener("wheel", (event) => {
  if (!event.ctrlKey) return;
  event.preventDefault();
  event.stopPropagation();
  zoomImageAtPointer(event.deltaY, event.clientX, event.clientY);
}, { passive: false });
$("#sidebarToggle").addEventListener("click", () => $("#itemSidebar").classList.toggle("open"));
$("#closeSidebar").addEventListener("click", () => $("#itemSidebar").classList.remove("open"));
$("#exportButton").addEventListener("click", exportResults);
$("#completionExportButton").addEventListener("click", exportResults);
$("#exitButton").addEventListener("click", showWelcome);
$("#completionExitButton").addEventListener("click", showWelcome);
$("#reviewButton").addEventListener("click", showWorkspace);
window.addEventListener("resize", applyZoom);
window.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !$("#appView").classList.contains("hidden")) {
    event.preventDefault();
    saveCompleteAndNext();
  }
});

refreshBootstrap().catch((error) => setStatus("#welcomeStatus", error.message, "error"));
