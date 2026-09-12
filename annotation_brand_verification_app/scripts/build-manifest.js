const fs = require("node:fs");
const path = require("node:path");

const APP_ROOT = path.resolve(__dirname, "..");
const DEFAULT_OUTPUT = path.join(APP_ROOT, "data", "generated_manifest.json");

const BRAND_CATEGORIES = [
  "Alcoholic drinks",
  "Automotive",
  "Business & industrial",
  "Clothing & accessories",
  "Financial services",
  "Food",
  "Household & domestic",
  "Leisure & entertainment",
  "Media & publishing",
  "Non-profit, public sector & education",
  "Pharma & healthcare",
  "Politics",
  "Retail",
  "Soft drinks",
  "Technology & electronics",
  "Telecoms & utilities",
  "Tobacco",
  "Toiletries & cosmetics",
  "Transport & tourism"
];

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8").replace(/^\uFEFF/, ""));
}

function requiredPath(name) {
  const value = String(process.env[name] || "").trim();
  if (!value) throw new Error(`Set ${name} to an input file.`);
  return path.resolve(value);
}

function publicMetadata(metadata = {}) {
  return Object.fromEntries(
    ["year", "decade"]
      .filter((key) => metadata[key] !== undefined && metadata[key] !== null)
      .map((key) => [key, metadata[key]])
  );
}

function buildManifest(exportedSet, modelRecords, createdAt = new Date().toISOString()) {
  const sourceImages = exportedSet?.annotation_set?.manifest?.images || exportedSet?.images;
  if (!Array.isArray(sourceImages) || !sourceImages.length) {
    throw new Error("Source set must contain annotation_set.manifest.images or images.");
  }

  const wanted = new Set(sourceImages.map((image) => image.image_id));
  const productionByImage = new Map();
  for (const record of modelRecords) {
    if (wanted.has(record.image_id)) productionByImage.set(record.image_id, record);
  }
  const missing = sourceImages.filter((image) => !productionByImage.has(image.image_id));
  if (missing.length) {
    throw new Error(`Missing model results for ${missing.length} source images.`);
  }

  let itemPosition = 0;
  const images = sourceImages.map((sourceImage, imageIndex) => {
    const record = productionByImage.get(sourceImage.image_id);
    const advertisements = (record.annotation?.advertisements || []).map((ad, adIndex) => {
      const advertisementId = String(ad.advertisement_id || `ad_${adIndex + 1}`);
      const item = {
        item_id: `${sourceImage.image_id}__${advertisementId}`,
        item_position: itemPosition,
        advertisement_id: advertisementId,
        bbox_1000: ad.bbox_1000,
        assigned_brand_category: ad.ad_category,
        assigned_brand_name: ad.brand_or_advertiser,
        extent: ad.extent || null,
        face_boxes_1000: (ad.people || [])
          .filter((person) => Array.isArray(person.face_bbox_1000) && person.face_bbox_1000.length === 4)
          .map((person) => ({
            person_id: person.person_id || null,
            bbox_1000: person.face_bbox_1000,
            confidence: person.confidence ?? null
          })),
        model_confidence: ad.confidence ?? null,
        model_review_flags: Array.isArray(ad.review_flags) ? ad.review_flags : []
      };
      itemPosition += 1;
      return item;
    });
    return {
      image_id: sourceImage.image_id,
      filename: path.basename(String(sourceImage.filename || `${sourceImage.image_id}.jpg`)),
      page_type: sourceImage.page_type || "single",
      image_position: imageIndex,
      metadata: publicMetadata(sourceImage.metadata),
      advertisements
    };
  });

  return {
    schema_version: "brand_verification_manifest_v1",
    task_id: String(process.env.BRAND_TASK_ID || "brand_verification"),
    name: String(process.env.BRAND_TASK_NAME || "Brand verification"),
    created_at: createdAt,
    source: { model_schema_version: "qwen_final_standalone_v1" },
    brand_categories: BRAND_CATEGORIES,
    image_count: images.length,
    item_count: itemPosition,
    images
  };
}

function readJsonl(file) {
  return fs.readFileSync(file, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function main() {
  const sourceSet = requiredPath("BRAND_SOURCE_SET");
  const modelResults = requiredPath("BRAND_MODEL_RESULTS");
  const output = path.resolve(process.env.BRAND_OUTPUT || DEFAULT_OUTPUT);
  const manifest = buildManifest(readJson(sourceSet), readJsonl(modelResults));
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  const pagesWithAds = manifest.images.filter((image) => image.advertisements.length).length;
  console.log(`Wrote ${output}`);
  console.log(`${manifest.images.length} images; ${pagesWithAds} pages with predicted ads; ${manifest.item_count} verification items.`);
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { BRAND_CATEGORIES, buildManifest, publicMetadata };
