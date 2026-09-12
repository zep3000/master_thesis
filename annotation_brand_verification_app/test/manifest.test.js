const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const manifest = require("../data/example_manifest.json");
const { buildManifest } = require("../scripts/build-manifest");

test("bundled manifest is a self-contained synthetic example", () => {
  const items = manifest.images.flatMap((image) => image.advertisements);
  assert.equal(manifest.schema_version, "brand_verification_manifest_v1");
  assert.equal(manifest.source.synthetic, true);
  assert.equal(manifest.images.length, 1);
  assert.equal(items.length, 1);
  assert.equal(path.isAbsolute(manifest.images[0].filename), false);
  assert.equal("path" in manifest.images[0], false);
});

test("manifest builder keeps model fields but removes machine paths and unrelated metadata", () => {
  const source = {
    images: [{
      image_id: "page-1",
      filename: "nested/page-1.jpg",
      page_type: "single",
      metadata: { year: 1980, decade: 1980, source_scan_id: "local-only", face_count: 3 }
    }]
  };
  const results = [{
    image_id: "page-1",
    annotation: {
      advertisements: [{
        advertisement_id: "ad_1",
        bbox_1000: [0, 0, 1000, 1000],
        ad_category: "Automotive",
        brand_or_advertiser: "Example Motors",
        people: []
      }]
    }
  }];
  const built = buildManifest(source, results, "2000-01-01T00:00:00.000Z");
  const image = built.images[0];
  assert.equal(image.filename, "page-1.jpg");
  assert.equal("path" in image, false);
  assert.deepEqual(image.metadata, { year: 1980, decade: 1980 });
  assert.equal(built.item_count, 1);
  assert.equal(image.advertisements[0].assigned_brand_name, "Example Motors");
});

test("manifest builder rejects missing model results", () => {
  assert.throws(
    () => buildManifest({ images: [{ image_id: "missing", filename: "missing.jpg" }] }, []),
    /Missing model results/
  );
});
