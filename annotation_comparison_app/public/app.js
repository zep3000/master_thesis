const Agreement = window.AnnotationAgreement;
const SOURCE_COLOR_COUNT = 48;
const AD_FIELDS = [
  ["Extent", "extent"],
  ["Depiction type", "depiction_type"],
  ["Face depiction count", "face_depiction_count_band"],
  ["Duplicate faces", "duplicate_faces_present"],
  ["Unique face identities", "unique_face_count"],
  ["Outstanding individuals", "has_outstanding_individuals"]
];
const PERSON_FIELDS = [
  ["Depiction type", "depiction_type"],
  ["Age", "perceived_age"],
  ["Gender presentation", "perceived_gender_presentation"],
  ["Expression legibility", "face_expression_legibility"],
  ["Orientation", "face_orientation"],
  ["Gaze target", "gaze_target"],
  ["Unboxed gaze target", "gaze_target_person_unboxed"],
  ["Mouth covered", "mouth_covered"],
  ["Mouth covering", "mouth_covering"],
  ["Mouth covering detail", "mouth_covering_other_text"],
  ["Smile", "smile_present"],
  ["Smile intensity", "smile_intensity"]
];
const GROUP_FIELDS = [
  ["Visual arrangement", "group_type"],
  ["Age composition", "age_composition"],
  ["Gender composition", "gender_presentation_composition"],
  ["Expression legibility", "expression_legibility_distribution"],
  ["Dominant gaze", "dominant_gaze"],
  ["Smile prevalence", "smile_prevalence"],
  ["Smile intensity", "dominant_smile_intensity"]
];

const LABEL_LANGUAGE_STORAGE_KEY = "annotationComparisonLabelLanguage";
const MANIFEST_STORAGE_KEY = "annotationComparisonManifestPath";
const IMAGE_DIR_STORAGE_KEY = "annotationComparisonImageDir";
const DISPLAY_LABELS_EN = {
  yes: "yes",
  no: "no",
  true: "yes",
  false: "no",
  partly: "partly",
  full_page: "full page",
  partial_page: "partial page",
  no_ads_on_page: "no ad(s) on this page",
  ads_present_no_visible_faces: "ads present, but none with visible faces",
  photo_of_person: "photo of person",
  illustration: "illustration",
  naturalistic_illustration: "illustration",
  stylized_illustration: "illustration",
  cartoon_or_caricature: "cartoon/caricature",
  generic_human_figure: "generic human figure",
  photo_of_artwork_or_statue: "photo of artwork/statue",
  drawing_of_statue_monument_or_public_symbol: "drawing of statue, monument, or public symbol",
  mask_mannequin_doll_or_puppet: "mask, mannequin, doll, or puppet",
  personified_object: "personified object",
  nonhuman_creature_with_face: "nonhuman creature with face",
  schematic_icon_or_logo_face: "schematic icon/logo face",
  multiple_types_present: "multiple types present",
  only_individuals: "fewer than 10 people",
  "10_20": "10-20",
  "20_plus": "20+",
  infant: "infant",
  child: "child",
  adolescent: "adolescent",
  young_adult: "young adult",
  middle_adult: "middle adult",
  older_adult: "older adult",
  feminine: "feminine",
  masculine: "masculine",
  ambiguous_or_androgynous: "ambiguous/androgynous",
  beyond_profile: "less than profile",
  profile: "profile",
  three_quarter: "three-quarter",
  frontal: "frontal",
  tilted_down: "legacy tilted down",
  tilted_up: "legacy tilted up",
  frontal_head_angled_down: "legacy frontal/head angled down",
  frontal_head_angled_up: "legacy frontal/head angled up",
  not_assessable: "not assessable",
  viewer_camera: "viewer/camera",
  another_person: "another person",
  advertised_product: "advertised product",
  other_object: "other object",
  off_frame_or_scene_direction: "scene direction/off-screen",
  eyes_covered: "eyes covered",
  closed_eyes: "closed eyes",
  hand: "hand",
  beard: "beard",
  other_body_part: "other body part",
  part_of_another_person: "(part of) another person",
  object: "object",
  object_in_mouth: "object in mouth",
  text_or_graphic_overlay: "text/graphic overlay",
  other: "other",
  "1_slight": "1 slight",
  "2_clear": "2 clear",
  "3_broad": "3 broad",
  "4_laughter_like": "4 laughter-like",
  "0_not_legible": "0 not legible",
  "1_low_legibility": "1 low legibility",
  "2_moderate_legibility": "2 moderate legibility",
  "3_high_legibility": "3 high legibility",
  interacting_group: "interacting group",
  posed_group: "posed group",
  audience: "audience",
  audience_or_crowd: "deprecated: audience/crowd - review",
  general_crowd: "other arrangement (legacy: general_crowd)",
  background_population: "people in the background",
  separate_portraits_or_composite: "separate portraits/composite",
  other_group: "other arrangement",
  young_only: "young only",
  middle_only: "middle only",
  older_only: "older only",
  mostly_young: "mostly young",
  mostly_middle: "mostly middle",
  mostly_older: "mostly older",
  mixed: "mixed",
  feminine_only: "feminine only",
  masculine_only: "masculine only",
  mostly_feminine: "mostly feminine",
  mostly_masculine: "mostly masculine",
  ambiguous_or_androgynous_present: "ambiguous/androgynous present",
  all_0_not_legible: "all not legible",
  mostly_0_not_legible: "mostly not legible",
  all_1_low_legibility: "all low legibility",
  mostly_1_low_legibility: "mostly low legibility",
  all_2_moderate_legibility: "all moderate legibility",
  mostly_2_moderate_legibility: "mostly moderate legibility",
  all_3_high_legibility: "all high legibility",
  mostly_3_high_legibility: "mostly high legibility",
  mixed_legibility: "mixed legibility",
  toward_viewer_camera: "toward viewer/camera",
  toward_each_other: "toward each other",
  toward_object: "toward object",
  none: "none",
  minority: "minority",
  about_half: "about half",
  majority: "majority",
  all: "all",
  slight: "slight",
  clear: "clear",
  broad_or_laughter_like: "broad/laughter-like"
};
const DISPLAY_LABELS_DE = {
  yes: "ja",
  no: "nein",
  true: "ja",
  false: "nein",
  partly: "teilweise",
  full_page: "ganzseitig",
  partial_page: "Teilseite",
  no_ads_on_page: "keine Anzeige(n) auf dieser Seite",
  ads_present_no_visible_faces: "Anzeigen vorhanden, aber keine mit sichtbaren Gesichtern",
  photo_of_person: "Foto einer Person",
  illustration: "Illustration",
  naturalistic_illustration: "Illustration",
  stylized_illustration: "Illustration",
  cartoon_or_caricature: "Cartoon/Karikatur",
  generic_human_figure: "generische menschliche Figur",
  photo_of_artwork_or_statue: "Foto von Kunstwerk/Statue",
  drawing_of_statue_monument_or_public_symbol: "Zeichnung von Statue, Denkmal oder oeffentlichem Symbol",
  mask_mannequin_doll_or_puppet: "Maske, Schaufensterpuppe, Puppe oder Marionette",
  personified_object: "personifiziertes Objekt",
  nonhuman_creature_with_face: "nichtmenschliches Wesen mit Gesicht",
  schematic_icon_or_logo_face: "schematisches Icon/Logo-Gesicht",
  multiple_types_present: "mehrere Typen vorhanden",
  only_individuals: "weniger als 10 Personen",
  "10_20": "10-20",
  "20_plus": "20+",
  infant: "Saeugling",
  child: "Kind",
  adolescent: "Jugendliche/r",
  young_adult: "junge/r Erwachsene/r",
  middle_adult: "mittleres Erwachsenenalter",
  older_adult: "aeltere/r Erwachsene/r",
  feminine: "feminin",
  masculine: "maskulin",
  ambiguous_or_androgynous: "ambig/androgyn",
  beyond_profile: "weniger als Profil",
  profile: "Profil",
  three_quarter: "Dreiviertel",
  frontal: "frontal",
  tilted_down: "Legacy: nach unten geneigt",
  tilted_up: "Legacy: nach oben geneigt",
  frontal_head_angled_down: "Legacy: frontal, Kopf nach unten geneigt",
  frontal_head_angled_up: "Legacy: frontal, Kopf nach oben geneigt",
  not_assessable: "nicht beurteilbar",
  viewer_camera: "Betrachter/Kamera",
  another_person: "andere Person",
  advertised_product: "beworbenes Produkt",
  other_object: "anderes Objekt",
  off_frame_or_scene_direction: "Szenenrichtung/ausserhalb des Bildes",
  eyes_covered: "Augen bedeckt",
  closed_eyes: "Augen geschlossen",
  hand: "Hand",
  beard: "Bart",
  other_body_part: "anderes Koerperteil",
  part_of_another_person: "(Teil einer) anderen Person",
  object: "Objekt",
  object_in_mouth: "Objekt im Mund",
  text_or_graphic_overlay: "Text/grafische Ueberlagerung",
  other: "anderes",
  "1_slight": "1 leicht",
  "2_clear": "2 deutlich",
  "3_broad": "3 breit",
  "4_laughter_like": "4 lachend",
  "0_not_legible": "0 nicht lesbar",
  "1_low_legibility": "1 geringe Lesbarkeit",
  "2_moderate_legibility": "2 mittlere Lesbarkeit",
  "3_high_legibility": "3 hohe Lesbarkeit",
  interacting_group: "interagierende Gruppe",
  posed_group: "gestellte Gruppe",
  audience: "Publikum",
  audience_or_crowd: "veraltet: Publikum/Menge - pruefen",
  general_crowd: "andere Anordnung (Legacy: general_crowd)",
  background_population: "Personen im Hintergrund",
  separate_portraits_or_composite: "getrennte Portraets/Komposit",
  other_group: "andere Anordnung",
  young_only: "nur jung",
  middle_only: "nur mittleres Alter",
  older_only: "nur aelter",
  mostly_young: "ueberwiegend jung",
  mostly_middle: "ueberwiegend mittleres Alter",
  mostly_older: "ueberwiegend aelter",
  mixed: "gemischt",
  feminine_only: "nur feminin",
  masculine_only: "nur maskulin",
  mostly_feminine: "ueberwiegend feminin",
  mostly_masculine: "ueberwiegend maskulin",
  ambiguous_or_androgynous_present: "ambig/androgyn vorhanden",
  all_0_not_legible: "alle nicht lesbar",
  mostly_0_not_legible: "ueberwiegend nicht lesbar",
  all_1_low_legibility: "alle geringe Lesbarkeit",
  mostly_1_low_legibility: "ueberwiegend geringe Lesbarkeit",
  all_2_moderate_legibility: "alle mittlere Lesbarkeit",
  mostly_2_moderate_legibility: "ueberwiegend mittlere Lesbarkeit",
  all_3_high_legibility: "alle hohe Lesbarkeit",
  mostly_3_high_legibility: "ueberwiegend hohe Lesbarkeit",
  mixed_legibility: "gemischte Lesbarkeit",
  toward_viewer_camera: "zum Betrachter/zur Kamera",
  toward_each_other: "zueinander",
  toward_object: "zu einem Objekt",
  none: "keine",
  minority: "Minderheit",
  about_half: "etwa die Haelfte",
  majority: "Mehrheit",
  all: "alle",
  slight: "leicht",
  clear: "deutlich",
  broad_or_laughter_like: "breit/lachend"
};

const state = {
  manifest: null,
  manifestOptions: [],
  selectedManifestPath: localStorage.getItem(MANIFEST_STORAGE_KEY) || "",
  imageDirOptions: [],
  selectedImageDir: localStorage.getItem(IMAGE_DIR_STORAGE_KEY) || "",
  sources: [],
  availability: {},
  records: [],
  manualMatches: [],
  manualMatchSelection: [],
  lastManualMatchClickAt: 0,
  currentIndex: 0,
  selectedSourceIds: new Set(),
  sourceColorAssignments: new Map(),
  activeTab: "summary",
  disagreementOnly: false,
  entities: new Set(["ad", "person", "group"]),
  zoom: 1,
  fitBaseWidth: 1,
  overlayPersonBoxes: [],
  naturalWidth: 1,
  naturalHeight: 1,
  labelLanguage: localStorage.getItem(LABEL_LANGUAGE_STORAGE_KEY) === "de" ? "de" : "en"
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayAnnotationValue(value) {
  if (value === null || value === undefined || value === "" || (Array.isArray(value) && !value.length)) return null;
  if (Array.isArray(value)) return value.map(displayAnnotationValue).filter((item) => item !== null).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  const key = String(value);
  const labels = state.labelLanguage === "de" ? DISPLAY_LABELS_DE : DISPLAY_LABELS_EN;
  return labels[key] || key.replaceAll("_", " ");
}

function flowSchemaVersion(record) {
  return String(
    record.annotation?.flow_source?.flow_schema_version ||
    record.annotation?.flow_schema_version ||
    record.metadata?.flow_schema_version ||
    record.metadata?.flow_version ||
    ""
  );
}

function compatibilityFindings(records) {
  const findings = [];
  for (const record of records) {
    const source = `${record.short_label} ${record.label}`;
    const version = flowSchemaVersion(record);
    if (version === "1.15") {
      findings.push({
        source,
        field: "flow_schema_version",
        value: version,
        note: "Accepted. Newer fields may be missing and should be treated as missing, not invalid."
      });
    }
    const page = record.annotation?.page || {};
    if (String(page.qualifying_ad_count) === "0" && !page.no_qualifying_ad_reason) {
      findings.push({
        source,
        field: "page.no_qualifying_ad_reason",
        value: "missing",
        note: "Old zero-face page needs backfill for no_ads_on_page vs ads_present_no_visible_faces."
      });
    }
    for (const ad of record.annotation?.advertisements || []) {
      for (const group of ad.groups || []) {
        if (group.group_type === "audience_or_crowd") {
          findings.push({
            source,
            field: "group_type",
            value: "audience_or_crowd",
            note: "Deprecated value. Review and recode to a current people-area arrangement."
          });
        }
        if (group.group_type === "general_crowd") {
          findings.push({
            source,
            field: "group_type",
            value: "general_crowd",
            note: "Compared as other_group for compatibility; raw value is preserved."
          });
        }
      }
    }
  }
  return findings;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

async function loadManifestOptions() {
  const data = await fetchJson("/api/manifests");
  state.manifestOptions = data.manifests || [];
  const hasSaved = state.selectedManifestPath &&
    state.manifestOptions.some((option) => option.path === state.selectedManifestPath);
  if (state.selectedManifestPath && !hasSaved) {
    state.selectedManifestPath = "";
    localStorage.removeItem(MANIFEST_STORAGE_KEY);
  }
  renderManifestSelect();
}

async function loadImageDirOptions() {
  const data = await fetchJson("/api/image-dirs");
  state.imageDirOptions = data.image_dirs || [];
  const hasSaved = state.selectedImageDir &&
    state.imageDirOptions.some((option) => option.path === state.selectedImageDir);
  if (state.selectedImageDir && !hasSaved) {
    state.selectedImageDir = "";
    localStorage.removeItem(IMAGE_DIR_STORAGE_KEY);
  }
  renderImageDirSelect();
}

function renderManifestSelect() {
  const select = $("#manifestSelect");
  if (!select) return;
  const options = state.manifestOptions.length
    ? state.manifestOptions
    : [{ label: "Auto: newest human export", path: "" }];
  select.innerHTML = options.map((option) =>
    `<option value="${escapeHtml(option.path)}">${escapeHtml(option.label)}</option>`
  ).join("");
  select.value = state.selectedManifestPath;
}

function renderImageDirSelect() {
  const select = $("#imageDirSelect");
  if (!select) return;
  const options = state.imageDirOptions.length
    ? state.imageDirOptions
    : [{ label: "Auto search", path: "" }];
  select.innerHTML = options.map((option) => {
    const suffix = option.exists === false ? " (missing)" : "";
    return `<option value="${escapeHtml(option.path)}">${escapeHtml(option.label + suffix)}</option>`;
  }).join("");
  select.value = state.selectedImageDir;
}

function generatedSourceColor(slot) {
  const hue = (slot * 137.508) % 360;
  const saturation = slot % 3 === 0 ? 72 : slot % 3 === 1 ? 64 : 78;
  const lightness = slot % 4 === 0 ? 34 : slot % 4 === 1 ? 42 : slot % 4 === 2 ? 30 : 48;
  return `hsl(${Math.round(hue)} ${saturation}% ${lightness}%)`;
}

function nextSourceColorSlot() {
  const used = new Set(state.sourceColorAssignments.values());
  for (let slot = 0; slot < SOURCE_COLOR_COUNT; slot += 1) {
    if (!used.has(slot)) return slot;
  }
  return state.sourceColorAssignments.size % SOURCE_COLOR_COUNT;
}

function ensureSourceColor(sourceId) {
  if (!sourceId) return generatedSourceColor(0);
  if (!state.sourceColorAssignments.has(sourceId)) {
    state.sourceColorAssignments.set(sourceId, nextSourceColorSlot());
  }
  return generatedSourceColor(state.sourceColorAssignments.get(sourceId));
}

function reconcileSourceColors() {
  for (const sourceId of [...state.sourceColorAssignments.keys()]) {
    if (!state.selectedSourceIds.has(sourceId)) state.sourceColorAssignments.delete(sourceId);
  }
  for (const sourceId of state.selectedSourceIds) ensureSourceColor(sourceId);
}

function sourceColor(sourceId) {
  return ensureSourceColor(sourceId);
}

function selectSource(sourceId) {
  state.selectedSourceIds.add(sourceId);
  ensureSourceColor(sourceId);
}

function deselectSource(sourceId) {
  state.selectedSourceIds.delete(sourceId);
  state.sourceColorAssignments.delete(sourceId);
}

function setSelectedSources(sourceIds) {
  state.selectedSourceIds = new Set(sourceIds);
  reconcileSourceColors();
}

function currentImage() {
  return state.manifest?.images[state.currentIndex] || null;
}

function selectedRecords() {
  return state.records.filter((record) => state.selectedSourceIds.has(record.source_id));
}

function availableSourceIds() {
  return new Set(state.records.map((record) => record.source_id));
}

function restoreSelection() {
  try {
    const saved = JSON.parse(localStorage.getItem("annotationComparisonSources") || "[]");
    if (Array.isArray(saved)) setSelectedSources(saved);
  } catch {}
  if (!state.selectedSourceIds.size) {
    const defaultSources = state.sources.slice(-6);
    setSelectedSources(defaultSources.map((source) => source.source_id));
  }
}

function persistSelection() {
  localStorage.setItem("annotationComparisonSources", JSON.stringify([...state.selectedSourceIds]));
}

function renderPageList() {
  const query = $("#pageSearch").value.trim().toLowerCase();
  const filter = $("#pageFilter").value;
  const selectedIds = [...state.selectedSourceIds];
  const selectedTotal = selectedIds.length;
  const html = state.manifest.images.map((image, index) => {
    const availability = state.availability[image.image_id] || { total: 0, human: 0, llm: 0 };
    const availableSelected = selectedIds.filter((sourceId) => (availability.source_ids || []).includes(sourceId)).length;
    const selectedCoverageClass = selectedTotal && availableSelected === selectedTotal
      ? "selected-full"
      : selectedTotal && availableSelected > 0
        ? "selected-partial"
        : selectedTotal
          ? "selected-none"
          : "";
    const matchesQuery = !query || image.filename.toLowerCase().includes(query);
    const matchesFilter = filter === "all" ||
      (filter === "multi" && availability.total >= 2) ||
      (filter === "human_llm" && availability.human > 0 && availability.llm > 0);
    if (!matchesQuery || !matchesFilter) return "";
    return `
      <button class="page-item ${index === state.currentIndex ? "active" : ""}" data-page-index="${index}" type="button">
        <span class="page-name">${escapeHtml(image.filename)}</span>
        <span class="source-count ${selectedCoverageClass}" title="${availableSelected} of ${selectedTotal} selected sources on this page">
          <b>${selectedTotal ? `S${availableSelected}/${selectedTotal}` : "S-"}</b><b>H${availability.human}</b><b>L${availability.llm}</b>
        </span>
      </button>`;
  }).join("");
  $("#pageList").innerHTML = html || '<div class="empty-state">No pages match this filter.</div>';
  $$("[data-page-index]").forEach((button) => button.addEventListener("click", () => loadPage(Number(button.dataset.pageIndex))));
}

function renderSourceList() {
  const available = availableSourceIds();
  const selected = state.sources.filter((source) => state.selectedSourceIds.has(source.source_id));
  const visible = selected.slice(0, 5);
  $("#sourceList").innerHTML = visible.map((source) => {
    const color = sourceColor(source.source_id);
    const hasPage = available.has(source.source_id);
    return `
      <span class="source-chip active" style="color:${color};opacity:${hasPage ? 1 : 0.48}" title="${escapeHtml(source.label)}">
        <span class="swatch"></span>
        <strong>${escapeHtml(source.short_label)}</strong>
        <span class="source-type">${source.type}${hasPage ? "" : " - no page"}</span>
      </span>`;
  }).join("") + (selected.length > visible.length ? `<span class="source-more">+${selected.length - visible.length} more</span>` : "");
  $("#selectedSourceCount").textContent = `${selected.length} of ${state.sources.length} selected`;
}

function renderSourceManager() {
  const query = $("#sourceSearch").value.trim().toLowerCase();
  const available = availableSourceIds();
  const groups = [["Human annotators", "human"], ["LLM runs", "llm"]];
  $("#sourceManagerList").innerHTML = groups.map(([title, type]) => {
    const sources = state.sources.filter((source) => source.type === type && (!query || `${source.label} ${source.path}`.toLowerCase().includes(query)));
    if (!sources.length) return "";
    return `<section class="source-group">
      <div class="source-group-title"><span>${title}</span><span>${sources.length}</span></div>
      ${sources.map((source) => {
        const hasPage = available.has(source.source_id);
        return `<label class="source-manager-row ${hasPage ? "" : "unavailable"}">
          <input type="checkbox" data-manager-source="${escapeHtml(source.source_id)}" ${state.selectedSourceIds.has(source.source_id) ? "checked" : ""}>
          <span class="swatch" style="color:${sourceColor(source.source_id)}"></span>
          <span class="source-manager-name"><strong>${escapeHtml(source.short_label)} - ${escapeHtml(source.label)}</strong><span>${escapeHtml(source.path)}</span></span>
          <span class="source-manager-coverage">${hasPage ? "on page" : "no page"}<br>${source.image_count} total</span>
        </label>`;
      }).join("")}
    </section>`;
  }).join("") || '<div class="empty-state">No sources match this search.</div>';
  $$('[data-manager-source]').forEach((input) => input.addEventListener("change", () => {
    if (input.checked) selectSource(input.dataset.managerSource);
    else deselectSource(input.dataset.managerSource);
    state.manualMatchSelection = [];
    persistSelection();
    renderPageList();
    renderComparison();
  }));
}

function openSourceManager() {
  renderSourceManager();
  $("#sourceDialog").showModal();
}

function renderLabelLanguageToggle() {
  $$("[data-label-language]").forEach((button) => {
    const selected = button.dataset.labelLanguage === state.labelLanguage;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function setStageSize() {
  if (!state.naturalWidth || !state.naturalHeight) return;
  const scroll = $("#imageScroll");
  const stage = $("#imageStage");
  const fitWidth = Math.max(280, scroll.clientWidth - 28);
  const fitHeight = Math.max(280, scroll.clientHeight - 28);
  if (!state.fitBaseWidth || state.fitBaseWidth <= 1) {
    const scale = Math.min(fitWidth / state.naturalWidth, fitHeight / state.naturalHeight, 1);
    state.fitBaseWidth = Math.max(260, Math.round(state.naturalWidth * scale));
  }
  const width = Math.round(state.fitBaseWidth * state.zoom);
  stage.style.width = `${width}px`;
  stage.style.aspectRatio = `${state.naturalWidth} / ${state.naturalHeight}`;
  $("#overlay").setAttribute("viewBox", `0 0 ${state.naturalWidth} ${state.naturalHeight}`);
}

function fitImage() {
  state.zoom = 1;
  state.fitBaseWidth = 1;
  setStageSize();
}

function zoomImage(delta, anchorEvent = null) {
  const scroll = $("#imageScroll");
  const stage = $("#imageStage");
  const previousRect = stage.getBoundingClientRect();
  const previousWidth = previousRect.width || state.fitBaseWidth;
  const previousHeight = previousRect.height || (state.naturalHeight && state.naturalWidth ? previousWidth * state.naturalHeight / state.naturalWidth : previousWidth);
  const previousZoom = state.zoom;
  state.zoom = Math.max(0.4, Math.min(3, state.zoom + delta));
  if (state.zoom === previousZoom) return;

  const rect = scroll.getBoundingClientRect();
  const anchorX = anchorEvent ? anchorEvent.clientX - rect.left : scroll.clientWidth / 2;
  const anchorY = anchorEvent ? anchorEvent.clientY - rect.top : scroll.clientHeight / 2;
  const imageX = previousWidth ? (scroll.scrollLeft + anchorX) / previousWidth : 0.5;
  const imageY = previousHeight ? (scroll.scrollTop + anchorY) / previousHeight : 0.5;

  setStageSize();

  const nextRect = stage.getBoundingClientRect();
  const nextWidth = nextRect.width || state.fitBaseWidth * state.zoom;
  const nextHeight = nextRect.height || (state.naturalHeight && state.naturalWidth ? nextWidth * state.naturalHeight / state.naturalWidth : nextWidth);
  scroll.scrollLeft = imageX * nextWidth - anchorX;
  scroll.scrollTop = imageY * nextHeight - anchorY;
}

function allBoxes(record) {
  const boxes = [];
  for (const ad of record.annotation?.advertisements || []) {
    if (state.entities.has("ad")) boxes.push({ source_id: record.source_id, kind: "ad", id: ad.ad_id, ad_id: ad.ad_id, entity_id: ad.ad_id, box: Agreement.normalizedBox(ad) });
    if (state.entities.has("person")) {
      for (const person of ad.people || []) boxes.push({ source_id: record.source_id, kind: "person", id: person.person_id, ad_id: ad.ad_id, entity_id: person.person_id, box: Agreement.normalizedBox(person) });
    }
    if (state.entities.has("group")) {
      for (const group of ad.groups || []) boxes.push({ source_id: record.source_id, kind: "group", id: group.group_id, ad_id: ad.ad_id, entity_id: group.group_id, box: Agreement.normalizedBox(group) });
    }
  }
  return boxes.filter((item) => item.box);
}

function svgElement(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function renderOverlay() {
  const overlay = $("#overlay");
  const stageRect = $("#imageStage").getBoundingClientRect();
  const minHitCssPx = 24;
  const minHitWidth = state.naturalWidth && stageRect.width ? minHitCssPx * state.naturalWidth / stageRect.width : 0;
  const minHitHeight = state.naturalHeight && stageRect.height ? minHitCssPx * state.naturalHeight / stageRect.height : 0;
  overlay.innerHTML = "";
  state.overlayPersonBoxes = [];
  for (const record of selectedRecords()) {
    const color = sourceColor(record.source_id);
    for (const item of allBoxes(record)) {
      const [x1, y1, x2, y2] = item.box;
      const imageX = x1 * state.naturalWidth;
      const imageY = y1 * state.naturalHeight;
      const imageWidth = (x2 - x1) * state.naturalWidth;
      const imageHeight = (y2 - y1) * state.naturalHeight;
      const selectedForManualMatch = item.kind === "person" && state.manualMatchSelection.some((candidate) => sameManualRef(candidate, item));
      if (item.kind === "person") {
        const hitWidth = Math.max(imageWidth, minHitWidth);
        const hitHeight = Math.max(imageHeight, minHitHeight);
        const hitX = Math.max(0, imageX - (hitWidth - imageWidth) / 2);
        const hitY = Math.max(0, imageY - (hitHeight - imageHeight) / 2);
        state.overlayPersonBoxes.push({
          ref: manualRefFromItem(item),
          actual: { x: imageX, y: imageY, width: imageWidth, height: imageHeight },
          hit: { x: hitX, y: hitY, width: hitWidth, height: hitHeight }
        });
        const hitRect = svgElement("rect", {
          x: hitX,
          y: hitY,
          width: Math.min(state.naturalWidth, hitWidth),
          height: Math.min(state.naturalHeight, hitHeight),
          class: "comparison-hitbox person"
        });
        hitRect.setAttribute("data-overlay-person", "true");
        hitRect.setAttribute("data-manual-source", item.source_id);
        hitRect.setAttribute("data-manual-ad", item.ad_id || "");
        hitRect.setAttribute("data-manual-entity", item.entity_id || "");
        hitRect.addEventListener("click", (event) => {
          event.stopPropagation();
          chooseOverlayPersonAt(event.clientX, event.clientY);
        });
        overlay.appendChild(hitRect);
      }
      const rect = svgElement("rect", {
        x: imageX,
        y: imageY,
        width: imageWidth,
        height: imageHeight,
        class: `comparison-box ${item.kind} ${selectedForManualMatch ? "selected-manual" : ""}`,
        stroke: color
      });
      overlay.appendChild(rect);
      const text = svgElement("text", {
        x: x1 * state.naturalWidth + 4,
        y: Math.max(16, y1 * state.naturalHeight - 5),
        class: "comparison-label",
        fill: color
      });
      text.textContent = `${record.short_label} ${item.kind[0].toUpperCase()} ${item.id}`;
      overlay.appendChild(text);
    }
  }
  $("#overlayLegend").innerHTML = selectedRecords().map((record) => `
    <span class="legend-item" style="color:${sourceColor(record.source_id)}"><span class="swatch"></span>${escapeHtml(record.short_label)} ${escapeHtml(record.label)}</span>
  `).join("") || '<span class="legend-item">Select a source with data for this page.</span>';
}

function distanceToRect(point, rect) {
  const dx = point.x < rect.x ? rect.x - point.x : point.x > rect.x + rect.width ? point.x - (rect.x + rect.width) : 0;
  const dy = point.y < rect.y ? rect.y - point.y : point.y > rect.y + rect.height ? point.y - (rect.y + rect.height) : 0;
  return Math.hypot(dx, dy);
}

function chooseOverlayPersonAt(clientX, clientY) {
  const overlayRect = $("#overlay").getBoundingClientRect();
  if (!overlayRect.width || !overlayRect.height) return false;
  const point = {
    x: (clientX - overlayRect.left) * state.naturalWidth / overlayRect.width,
    y: (clientY - overlayRect.top) * state.naturalHeight / overlayRect.height
  };
  let candidates = state.overlayPersonBoxes.filter((entry) =>
    point.x >= entry.hit.x &&
    point.x <= entry.hit.x + entry.hit.width &&
    point.y >= entry.hit.y &&
    point.y <= entry.hit.y + entry.hit.height
  );
  if (!candidates.length) return false;

  const selected = state.manualMatchSelection[0];
  if (selected) {
    const differentSource = candidates.filter((entry) => entry.ref.source_id !== selected.source_id);
    if (differentSource.length) candidates = differentSource;
    const differentEntity = candidates.filter((entry) => !sameManualRef(entry.ref, selected));
    if (differentEntity.length) candidates = differentEntity;
  }

  candidates.sort((a, b) => {
    const aActualDistance = distanceToRect(point, a.actual);
    const bActualDistance = distanceToRect(point, b.actual);
    if (aActualDistance !== bActualDistance) return aActualDistance - bActualDistance;
    const aArea = a.actual.width * a.actual.height;
    const bArea = b.actual.width * b.actual.height;
    return aArea - bArea;
  });
  chooseManualMatchRef(candidates[0].ref);
  return true;
}

function filterRows(rows) {
  return state.disagreementOnly ? rows.filter((row) => row.comparison.state === "disagree") : rows;
}

function sourceById(sourceId) {
  return state.sources.find((source) => source.source_id === sourceId) || null;
}

function manualRefFromItem(item) {
  return {
    source_id: item.source_id,
    ad_id: item.ad_id || "",
    entity_id: item.entity_id || ""
  };
}

function sameManualRef(a, b) {
  return Agreement.entityRefKey(a) === Agreement.entityRefKey(b);
}

function manualRefLabel(ref) {
  const source = sourceById(ref.source_id);
  return `${source?.short_label || ref.source_id} ${ref.entity_id}`;
}

function manualMatchControls(cluster, kind) {
  if (kind !== "person") return "";
  const clusterKeys = new Set(cluster.items.map((item) => Agreement.entityRefKey(item)));
  const clusterMatches = state.manualMatches.filter((match) =>
    match.kind === kind &&
    clusterKeys.has(Agreement.entityRefKey(match.left)) &&
    clusterKeys.has(Agreement.entityRefKey(match.right))
  );
  const badge = clusterMatches.length
    ? '<span class="manual-match-badge">manual</span>'
    : "";
  const buttons = cluster.items.map((item) => {
    const ref = manualRefFromItem(item);
    const selected = state.manualMatchSelection.some((candidate) => sameManualRef(candidate, ref));
    return `<button type="button" class="manual-match-chip ${selected ? "selected" : ""}" data-manual-source="${escapeHtml(ref.source_id)}" data-manual-ad="${escapeHtml(ref.ad_id)}" data-manual-entity="${escapeHtml(ref.entity_id)}">${escapeHtml(manualRefLabel(ref))}</button>`;
  }).join("");
  const selection = state.manualMatchSelection.length
    ? `<span class="manual-match-selection">Selected: ${escapeHtml(state.manualMatchSelection.map(manualRefLabel).join(" + "))}</span>`
    : '<span class="manual-match-selection">Select two person boxes from different sources to force a match.</span>';
  const removals = clusterMatches.map((match) =>
    `<button type="button" class="manual-match-remove" data-manual-remove="${escapeHtml(match.id)}">Remove ${escapeHtml(manualRefLabel(match.left))} + ${escapeHtml(manualRefLabel(match.right))}</button>`
  ).join("");
  return `<div class="manual-match-tools">${badge}<div class="manual-match-chips">${buttons}</div>${selection}${removals ? `<div class="manual-match-removals">${removals}</div>` : ""}</div>`;
}

async function saveManualMatch(left, right) {
  const image = currentImage();
  const response = await fetchJson("/api/manual-matches", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ image_id: image.image_id, kind: "person", left, right })
  });
  state.manualMatches = response.matches || [];
  state.manualMatchSelection = [];
  renderComparison();
  $("#loadStatus").textContent = `Saved manual match: ${manualRefLabel(left)} + ${manualRefLabel(right)}.`;
}

function chooseManualMatchRef(ref) {
  const now = performance.now();
  if (now - state.lastManualMatchClickAt < 220) return;
  state.lastManualMatchClickAt = now;
  const first = state.manualMatchSelection[0];
  if (!first) {
    state.manualMatchSelection = [ref];
    renderComparison();
    $("#loadStatus").textContent = `Selected ${manualRefLabel(ref)}. Click a person box from another source to save a match.`;
    return;
  }
  if (sameManualRef(first, ref)) {
    state.manualMatchSelection = [];
    renderComparison();
    $("#loadStatus").textContent = "Manual match selection cleared.";
    return;
  }
  if (first.source_id === ref.source_id) {
    state.manualMatchSelection = [ref];
    renderComparison();
    $("#loadStatus").textContent = `Selected ${manualRefLabel(ref)}. Manual matches must connect two different sources.`;
    return;
  }
  saveManualMatch(first, ref).catch((error) => {
    $("#loadStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
  });
}

function bindManualMatchButtons() {
  $$("[data-manual-source]").forEach((button) => button.addEventListener("click", () => {
    chooseManualMatchRef({
      source_id: button.dataset.manualSource,
      ad_id: button.dataset.manualAd,
      entity_id: button.dataset.manualEntity
    });
  }));
  $$("[data-manual-remove]").forEach((button) => button.addEventListener("click", async () => {
    try {
      const response = await fetchJson("/api/manual-matches", {
        method: "DELETE",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ image_id: currentImage().image_id, id: button.dataset.manualRemove })
      });
      state.manualMatches = response.matches || [];
      state.manualMatchSelection = [];
      renderComparison();
    } catch (error) {
      $("#loadStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
    }
  }));
}

function matrixHtml(rows, sourceIds) {
  const sourceMap = new Map(state.sources.map((source) => [source.source_id, source]));
  const visibleRows = filterRows(rows);
  if (!visibleRows.length) return '<div class="empty-state">No disagreements in this section.</div>';
  return `
    <div class="matrix-wrap"><table class="comparison-matrix">
      <thead><tr><th>Field</th>${sourceIds.map((id) => `<th style="color:${sourceColor(id)}">${escapeHtml(sourceMap.get(id)?.short_label || id)}</th>`).join("")}</tr></thead>
      <tbody>${visibleRows.map((row) => `
        <tr class="row-${row.comparison.state}">
          <th class="field-label"><span class="state-dot"></span>${escapeHtml(row.label)}</th>
          ${sourceIds.map((id) => {
            const value = displayAnnotationValue(row.values[id]);
            return `<td class="${value === null ? "cell-missing" : ""}">${value === null ? "missing" : escapeHtml(value)}</td>`;
          }).join("")}
        </tr>`).join("")}</tbody>
    </table></div>`;
}

function sectionHtml(title, rows, sourceIds, spatial = null, toolsHtml = "") {
  if (state.disagreementOnly && !rows.some((row) => row.comparison.state === "disagree")) return "";
  const spatialText = spatial === null ? "" : `Mean box IoU ${Math.round(spatial * 100)}%`;
  return `<section class="comparison-section"><div class="section-heading"><h3>${escapeHtml(title)}</h3><span class="spatial-badge">${spatialText}</span></div>${toolsHtml}${matrixHtml(rows, sourceIds)}</section>`;
}

function compatibilityHtml(records) {
  const findings = compatibilityFindings(records);
  if (!findings.length) return "";
  return `<section class="compatibility-note">
    <div class="section-heading"><h3>Compatibility notes</h3></div>
    <table>
      <thead><tr><th>Source</th><th>Field</th><th>Value</th><th>Handling</th></tr></thead>
      <tbody>${findings.map((finding) => `<tr>
        <td>${escapeHtml(finding.source)}</td>
        <td>${escapeHtml(finding.field)}</td>
        <td>${escapeHtml(finding.value)}</td>
        <td>${escapeHtml(finding.note)}</td>
      </tr>`).join("")}</tbody>
    </table>
  </section>`;
}

function comparisonSections(records, sourceIds) {
  const pageRows = Agreement.pageRows(records, sourceIds);
  const ads = Agreement.clusterEntities(records, "ad", 0.2);
  const people = Agreement.clusterEntities(records, "person", 0.2, state.manualMatches);
  const groups = Agreement.clusterEntities(records, "group", 0.18);
  const sections = {
    summary: `${compatibilityHtml(records)}${sectionHtml("Page", pageRows, sourceIds)}`,
    ads: ads.map((cluster, index) => sectionHtml(`Ad match ${index + 1}`, Agreement.clusterRows(cluster, sourceIds, AD_FIELDS), sourceIds, cluster.mean_iou)).join(""),
    people: people.map((cluster, index) => sectionHtml(`Person match ${index + 1}`, Agreement.clusterRows(cluster, sourceIds, PERSON_FIELDS), sourceIds, cluster.mean_iou, manualMatchControls(cluster, "person"))).join(""),
    groups: groups.map((cluster, index) => sectionHtml(`People-area match ${index + 1}`, Agreement.clusterRows(cluster, sourceIds, GROUP_FIELDS), sourceIds, cluster.mean_iou)).join("")
  };
  const allRows = [
    ...pageRows,
    ...ads.flatMap((cluster) => Agreement.clusterRows(cluster, sourceIds, AD_FIELDS)),
    ...people.flatMap((cluster) => Agreement.clusterRows(cluster, sourceIds, PERSON_FIELDS)),
    ...groups.flatMap((cluster) => Agreement.clusterRows(cluster, sourceIds, GROUP_FIELDS))
  ];
  return { sections, summary: Agreement.summarizeRows(allRows), clusters: { ads, people, groups } };
}

function provenanceHtml(records) {
  if (!records.length) return '<div class="empty-state">No selected sources have a record for this page.</div>';
  return `<section class="comparison-section provenance-list">${records.map((record) => {
    const metadata = record.metadata || {};
    return `<div class="provenance-row">
      <strong style="color:${sourceColor(record.source_id)}">${escapeHtml(record.short_label)} ${escapeHtml(record.label)}</strong>
      <dl>
        <dt>Type</dt><dd>${escapeHtml(record.type)}</dd>
        <dt>Status</dt><dd>${escapeHtml(metadata.status || record.annotation?.status || (metadata.ok ? "ok" : ""))}</dd>
        <dt>Model</dt><dd>${escapeHtml(metadata.model || "")}</dd>
        <dt>Updated</dt><dd>${escapeHtml(metadata.processed_at || metadata.updated_at || "")}</dd>
        <dt>Source</dt><dd>${escapeHtml(metadata.source_file || "")}</dd>
        <dt>Validation</dt><dd>${escapeHtml((metadata.validation_errors || []).join("; ") || "none")}</dd>
        <dt>Normalization</dt><dd>${escapeHtml((metadata.normalization_actions || []).join("; ") || "none")}</dd>
      </dl>
    </div>`;
  }).join("")}</section>`;
}

function renderAgreementStrip(summary, records) {
  const score = summary.score === null ? "n/a" : `${Math.round(summary.score * 100)}%`;
  $("#agreementStrip").innerHTML = `
    <div class="metric"><span>Sources on page</span><strong>${records.length}</strong></div>
    <div class="metric agree"><span>Agreement</span><strong>${score}</strong></div>
    <div class="metric disagree"><span>Disagreements</span><strong>${summary.disagree}</strong></div>
    <div class="metric"><span>Comparable fields</span><strong>${summary.comparable}</strong></div>`;
}

function renderComparison() {
  renderLabelLanguageToggle();
  renderSourceList();
  if ($("#sourceDialog").open) renderSourceManager();
  renderOverlay();
  const records = selectedRecords();
  const sourceIds = state.sources.filter((source) => state.selectedSourceIds.has(source.source_id)).map((source) => source.source_id);
  const comparison = comparisonSections(records, sourceIds);
  renderAgreementStrip(comparison.summary, records);
  let html = state.activeTab === "provenance" ? provenanceHtml(records) : comparison.sections[state.activeTab];
  if (!records.length) {
    const coverage = state.availability[currentImage()?.image_id] || { total: 0 };
    html = `<div class="coverage-empty"><strong>No annotation result exists for this page yet.</strong><br>The current source scan reports ${coverage.total} records. Refresh after a human or LLM run has processed this page.</div>`;
  } else if (!html) {
    html = '<div class="empty-state">No comparable annotations in this view. Select additional sources or another page.</div>';
  }
  $("#comparisonContent").innerHTML = html;
  bindManualMatchButtons();
  $("#selectedSourceCount").textContent = `${state.selectedSourceIds.size} of ${state.sources.length} selected`;
}

async function loadPage(index) {
  if (!state.manifest) return;
  state.currentIndex = Math.max(0, Math.min(index, state.manifest.images.length - 1));
  const image = currentImage();
  $("#loadStatus").textContent = `Loading ${image.filename}...`;
  $("#prevPage").disabled = state.currentIndex === 0;
  $("#nextPage").disabled = state.currentIndex === state.manifest.images.length - 1;
  $("#pagePosition").textContent = `${state.currentIndex + 1} / ${state.manifest.images.length}`;
  $("#imageTitle").textContent = image.filename;
  renderPageList();
  state.manualMatchSelection = [];
  const [response, manualResponse] = await Promise.all([
    fetchJson(`/api/comparison?image_id=${encodeURIComponent(image.image_id)}`),
    fetchJson(`/api/manual-matches?image_id=${encodeURIComponent(image.image_id)}`)
  ]);
  state.records = response.records;
  state.manualMatches = manualResponse.matches || [];
  const pageAvailable = availableSourceIds();
  if (![...state.selectedSourceIds].some((id) => pageAvailable.has(id)) && pageAvailable.size) {
    setSelectedSources([...pageAvailable].slice(0, 6));
    persistSelection();
    renderPageList();
  }
  $("#imageSourceCount").textContent = `${state.records.length} annotation source${state.records.length === 1 ? "" : "s"}`;
  const pageImage = $("#pageImage");
  pageImage.onload = () => {
    state.naturalWidth = pageImage.naturalWidth || 1;
    state.naturalHeight = pageImage.naturalHeight || 1;
    state.zoom = 1;
    state.fitBaseWidth = 1;
    setStageSize();
    renderOverlay();
  };
  pageImage.src = `/api/image?path=${encodeURIComponent(image.path)}`;
  renderComparison();
  $("#loadStatus").textContent = `${state.records.length} records loaded for ${image.filename}`;
}

function bindEvents() {
  $("#prevPage").addEventListener("click", () => loadPage(state.currentIndex - 1));
  $("#nextPage").addEventListener("click", () => loadPage(state.currentIndex + 1));
  $("#togglePages").addEventListener("click", () => $("#workspace").classList.toggle("pages-open"));
  $("#toggleSources").addEventListener("click", openSourceManager);
  $("#manageSources").addEventListener("click", openSourceManager);
  $("#closeSourceDialog").addEventListener("click", () => $("#sourceDialog").close());
  $("#sourceSearch").addEventListener("input", renderSourceManager);
  $("#pageSearch").addEventListener("input", renderPageList);
  $("#pageFilter").addEventListener("change", renderPageList);
  $$("[data-source-action]").forEach((button) => button.addEventListener("click", () => {
    const action = button.dataset.sourceAction;
    if (action === "available") setSelectedSources(availableSourceIds());
    if (action === "human") setSelectedSources(state.sources.filter((source) => source.type === "human").map((source) => source.source_id));
    if (action === "llm") setSelectedSources(state.sources.filter((source) => source.type === "llm").map((source) => source.source_id));
    if (action === "all") setSelectedSources(state.sources.map((source) => source.source_id));
    if (action === "clear") setSelectedSources([]);
    state.manualMatchSelection = [];
    persistSelection();
    renderPageList();
    renderComparison();
  }));
  $("#refreshData").addEventListener("click", () => {
    Promise.all([loadManifestOptions(), loadImageDirOptions()])
      .then(() => refreshData(false))
      .catch((error) => {
        $("#loadStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      });
  });
  $("#manifestSelect").addEventListener("change", (event) => {
    state.selectedManifestPath = event.target.value;
    if (state.selectedManifestPath) localStorage.setItem(MANIFEST_STORAGE_KEY, state.selectedManifestPath);
    else localStorage.removeItem(MANIFEST_STORAGE_KEY);
    state.currentIndex = 0;
    refreshData(false).catch((error) => {
      $("#loadStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
    });
  });
  $("#imageDirSelect").addEventListener("change", (event) => {
    state.selectedImageDir = event.target.value;
    if (state.selectedImageDir) localStorage.setItem(IMAGE_DIR_STORAGE_KEY, state.selectedImageDir);
    else localStorage.removeItem(IMAGE_DIR_STORAGE_KEY);
    refreshData(false).catch((error) => {
      $("#loadStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
    });
  });
  $$("[data-tab]").forEach((button) => button.addEventListener("click", () => {
    state.activeTab = button.dataset.tab;
    $$("[data-tab]").forEach((item) => item.classList.toggle("active", item === button));
    renderComparison();
  }));
  $("#disagreementOnly").addEventListener("change", (event) => {
    state.disagreementOnly = event.target.checked;
    renderComparison();
  });
  $$("[data-label-language]").forEach((button) => button.addEventListener("click", () => {
    state.labelLanguage = button.dataset.labelLanguage === "de" ? "de" : "en";
    localStorage.setItem(LABEL_LANGUAGE_STORAGE_KEY, state.labelLanguage);
    renderComparison();
  }));
  $$("[data-entity]").forEach((input) => input.addEventListener("change", () => {
    if (input.checked) state.entities.add(input.dataset.entity);
    else state.entities.delete(input.dataset.entity);
    if (input.dataset.entity === "person" && !input.checked) state.manualMatchSelection = [];
    renderOverlay();
  }));
  $$("[data-zoom]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.zoom === "fit") return fitImage();
    if (button.dataset.zoom === "in") zoomImage(0.2);
    if (button.dataset.zoom === "out") zoomImage(-0.2);
  }));
  $("#imageScroll").addEventListener("wheel", (event) => {
    if (!event.ctrlKey) return;
    event.preventDefault();
    zoomImage(event.deltaY < 0 ? 0.2 : -0.2, event);
  }, { passive: false });
  $("#overlay").addEventListener("click", (event) => {
    if (!chooseOverlayPersonAt(event.clientX, event.clientY) && state.manualMatchSelection.length) {
      state.manualMatchSelection = [];
      renderComparison();
      $("#loadStatus").textContent = "Manual match selection cleared.";
    }
  });
  $("#imageStage").addEventListener("click", () => {
    if (!state.manualMatchSelection.length) return;
    state.manualMatchSelection = [];
    renderComparison();
    $("#loadStatus").textContent = "Manual match selection cleared.";
  });
  window.addEventListener("resize", () => {
    state.fitBaseWidth = 1;
    setStageSize();
  });
}

async function refreshData(initial) {
  $("#loadStatus").textContent = "Scanning annotation sources...";
  state.manualMatchSelection = [];
  const currentId = currentImage()?.image_id;
  const previousIds = new Set(state.sources.map((source) => source.source_id));
  const params = new URLSearchParams();
  if (state.selectedManifestPath) params.set("manifest", state.selectedManifestPath);
  if (state.selectedImageDir) params.set("image_dir", state.selectedImageDir);
  const query = params.toString() ? `?${params}` : "";
  const data = await fetchJson(`/api/bootstrap${query}`);
  state.manifest = data.manifest;
  state.sources = data.sources;
  state.availability = data.availability;
  if (initial) restoreSelection();
  else {
    for (const source of state.sources.filter((item) => !previousIds.has(item.source_id))) {
      selectSource(source.source_id);
    }
  }
  reconcileSourceColors();
  if (currentId) {
    const found = state.manifest.images.findIndex((image) => image.image_id === currentId);
    if (found >= 0) state.currentIndex = found;
  }
  $("#taskLabel").textContent = `${state.manifest.task_id} - ${state.sources.length} discovered sources`;
  renderPageList();
  await loadPage(state.currentIndex);
}

bindEvents();
Promise.all([loadManifestOptions(), loadImageDirOptions()])
  .then(() => refreshData(true))
  .catch((error) => {
    $("#loadStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
  });
