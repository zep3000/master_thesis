const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const {
  completionErrors,
  flattenItems,
  normalizeAnnotation,
  resolveImagePath,
  safeItemFilename,
  validSessionId
} = require("../server");

const categories = ["Automotive", "Food"];
const item = {
  item_id: "page-1__ad_1",
  image_id: "page-1",
  filename: "page-1.jpg",
  advertisement_id: "ad_1",
  item_position: 0,
  bbox_1000: [0, 0, 1000, 1000],
  assigned_brand_category: "Automotive",
  assigned_brand_name: "Example Motors"
};

test("session and item identifiers are constrained", () => {
  assert.equal(validSessionId("12345678"), "12345678");
  assert.equal(validSessionId("../12345678"), null);
  assert.equal(safeItemFilename("page/one__ad:1"), "page_one__ad_1.json");
});

test("image paths use the configured root and manifest basename", () => {
  const imageRoot = path.resolve("test-images");
  assert.equal(
    resolveImagePath({ filename: "../example-page.jpg", path: "ignored/other-page.jpg" }, imageRoot),
    path.join(imageRoot, "example-page.jpg")
  );
});

test("flattenItems turns multi-ad pages into separate verification items", () => {
  const result = flattenItems({ images: [{ image_id: "p", filename: "p.jpg", image_position: 0, metadata: { year: 1980 }, advertisements: [{ item_id: "p__a1" }, { item_id: "p__a2" }] }] });
  assert.equal(result.length, 2);
  assert.equal(result[1].image_id, "p");
  assert.equal(result[1].image_metadata.year, 1980);
});

test("complete annotations preserve originals and derive effective values", () => {
  const annotation = normalizeAnnotation(item, {
    category_review: "incorrect",
    corrected_brand_category: "Food",
    brand_name_review: "correct"
  }, categories, true);
  assert.equal(annotation.status, "complete");
  assert.equal(annotation.original.brand_category, "Automotive");
  assert.equal(annotation.effective_brand_category, "Food");
  assert.equal(annotation.effective_brand_name, "Example Motors");
});

test("incorrect answers require valid corrections", () => {
  assert.throws(() => normalizeAnnotation(item, {
    category_review: "incorrect",
    corrected_brand_category: "Unknown",
    brand_name_review: "incorrect",
    corrected_brand_name: ""
  }, categories, true), /Choose the correct brand category/);
});

test("wrong identification can be completed without category or brand verdicts", () => {
  const annotation = normalizeAnnotation(item, { identification_wrong: true, identification_note: "Not an ad" }, categories, true);
  assert.equal(annotation.status, "complete");
  assert.equal(annotation.effective_brand_category, null);
  assert.equal(annotation.effective_brand_name, null);
  assert.deepEqual(completionErrors(annotation, categories), []);
});

test("no-faces review is optional and is preserved independently", () => {
  const annotation = normalizeAnnotation(item, {
    no_faces: true,
    category_review: "correct",
    brand_name_review: "correct"
  }, categories, true);
  assert.equal(annotation.status, "complete");
  assert.equal(annotation.no_faces, true);
  assert.equal(annotation.effective_brand_name, "Example Motors");
});
