const test = require("node:test");
const assert = require("node:assert/strict");

const { imageIdFromRecord, isComparableAnnotation } = require("../server");

test("record matching prefers filename over collection-specific ids and paths", () => {
  const record = {
    image: {
      image_id: "clean-collection-internal-id",
      filename: "collection/example-page-001.JPG",
      path: "elsewhere/example-page-001.JPG"
    }
  };

  assert.equal(imageIdFromRecord(record), "example-page-001");
});

test("record matching falls back to an id only when no filename is available", () => {
  assert.equal(imageIdFromRecord({ image_id: "PAGE-42" }), "page-42");
});

test("1.15 annotations are accepted by structure, not rejected by version", () => {
  assert.equal(isComparableAnnotation({
    flow_source: { flow_schema_version: "1.15" },
    page: { qualifying_ad_count: "0" },
    advertisements: []
  }), true);
});
