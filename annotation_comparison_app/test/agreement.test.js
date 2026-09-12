const test = require("node:test");
const assert = require("node:assert/strict");
const Agreement = require("../public/agreement.js");

test("box IoU handles exact and disjoint boxes", () => {
  assert.equal(Agreement.boxIou([0, 0, 1, 1], [0, 0, 1, 1]), 1);
  assert.equal(Agreement.boxIou([0, 0, 0.2, 0.2], [0.8, 0.8, 1, 1]), 0);
});

test("entities from different sources cluster by overlap", () => {
  const records = [
    { source_id: "a", annotation: { advertisements: [{ ad_id: "ad1", bbox: [0, 0, 1, 1], people: [{ person_id: "p1", face_bbox: [0.1, 0.1, 0.2, 0.2] }], groups: [] }] } },
    { source_id: "b", annotation: { advertisements: [{ ad_id: "ad1", bbox: [0, 0, 1, 1], people: [{ person_id: "p1", face_bbox: [0.11, 0.1, 0.21, 0.2] }], groups: [] }] } }
  ];
  const clusters = Agreement.clusterEntities(records, "person", 0.2);
  assert.equal(clusters.length, 1);
  assert.equal(clusters[0].items.length, 2);
  assert.ok(clusters[0].mean_iou > 0.7);
});

test("manual person matches override missing box overlap", () => {
  const records = [
    { source_id: "a", annotation: { advertisements: [{ ad_id: "ad1", bbox: [0, 0, 1, 1], people: [{ person_id: "p1", face_bbox: [0.1, 0.1, 0.2, 0.2] }], groups: [] }] } },
    { source_id: "b", annotation: { advertisements: [{ ad_id: "ad1", bbox: [0, 0, 1, 1], people: [{ person_id: "p9", face_bbox: [0.8, 0.8, 0.9, 0.9] }], groups: [] }] } }
  ];
  const manualMatches = [{
    image_id: "page-1",
    kind: "person",
    left: { source_id: "a", ad_id: "ad1", entity_id: "p1" },
    right: { source_id: "b", ad_id: "ad1", entity_id: "p9" }
  }];
  const clusters = Agreement.clusterEntities(records, "person", 0.2, manualMatches);
  assert.equal(clusters.length, 1);
  assert.equal(clusters[0].items.length, 2);
  assert.equal(clusters[0].manual_match_count, 1);
});

test("person rows inherit ad-level depiction type when person value is absent", () => {
  const records = [
    {
      source_id: "a",
      annotation: {
        advertisements: [{
          ad_id: "ad1",
          depiction_type: "photo_of_person",
          people: [{ person_id: "p1", face_bbox: [0.1, 0.1, 0.2, 0.2] }],
          groups: []
        }]
      }
    }
  ];
  const [cluster] = Agreement.clusterEntities(records, "person", 0.2);
  const [row] = Agreement.clusterRows(cluster, ["a"], [["Depiction type", "depiction_type"]]);
  assert.equal(row.values.a, "photo_of_person");
});

test("person-specific depiction type wins over ad-level depiction type", () => {
  const records = [
    {
      source_id: "a",
      annotation: {
        advertisements: [{
          ad_id: "ad1",
          depiction_type: "multiple_types_present",
          people: [{ person_id: "p1", depiction_type: "cartoon_or_caricature", face_bbox: [0.1, 0.1, 0.2, 0.2] }],
          groups: []
        }]
      }
    }
  ];
  const [cluster] = Agreement.clusterEntities(records, "person", 0.2);
  const [row] = Agreement.clusterRows(cluster, ["a"], [["Depiction type", "depiction_type"]]);
  assert.equal(row.values.a, "cartoon_or_caricature");
});

test("agreement distinguishes consensus, disagreement, and missing", () => {
  assert.equal(Agreement.compareValues({ a: "yes", b: "yes" }, ["a", "b"]).state, "agree");
  assert.equal(Agreement.compareValues({ a: "yes", b: "no" }, ["a", "b"]).state, "disagree");
  assert.equal(Agreement.compareValues({ a: "yes" }, ["a", "b"]).state, "insufficient");
});

test("comparison normalizes safely migrated legacy values", () => {
  assert.equal(Agreement.compareValues({ a: "general_crowd", b: "other_group" }, ["a", "b"]).state, "agree");
  assert.equal(Agreement.compareValues({ a: "off_frame", b: "off_frame_or_scene_direction" }, ["a", "b"]).state, "agree");
  assert.equal(Agreement.compareValues({ a: "audience_or_crowd", b: "audience" }, ["a", "b"]).state, "disagree");
});
