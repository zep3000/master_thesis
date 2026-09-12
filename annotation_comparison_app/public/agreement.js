(function expose(root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.AnnotationAgreement = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function factory() {
  function normalizedBox(entity) {
    const box = entity?.bbox || entity?.face_bbox;
    if (Array.isArray(box) && box.length === 4) return box.map(Number);
    const box1000 = entity?.bbox_1000 || entity?.face_bbox_1000;
    if (Array.isArray(box1000) && box1000.length === 4) return box1000.map((value) => Number(value) / 1000);
    return null;
  }

  function boxIou(a, b) {
    if (!a || !b) return 0;
    const ix1 = Math.max(a[0], b[0]);
    const iy1 = Math.max(a[1], b[1]);
    const ix2 = Math.min(a[2], b[2]);
    const iy2 = Math.min(a[3], b[3]);
    const intersection = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    const union = areaA + areaB - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function flattenEntities(records, kind) {
    const output = [];
    for (const record of records) {
      const ads = record.annotation?.advertisements || [];
      if (kind === "ad") {
        for (const entity of ads) output.push({ source_id: record.source_id, ad_id: entity.ad_id, entity_id: entity.ad_id, entity, box: normalizedBox(entity) });
      }
      if (kind === "person") {
        for (const ad of ads) {
          for (const entity of ad.people || []) {
            const displayEntity = entity.depiction_type
              ? entity
              : { ...entity, depiction_type: ad.depiction_type ?? entity.depiction_type };
            output.push({ source_id: record.source_id, ad_id: ad.ad_id, entity_id: entity.person_id, entity: displayEntity, box: normalizedBox(entity) });
          }
        }
      }
      if (kind === "group") {
        for (const ad of ads) {
          for (const entity of ad.groups || []) output.push({ source_id: record.source_id, ad_id: ad.ad_id, entity_id: entity.group_id, entity, box: normalizedBox(entity) });
        }
      }
    }
    return output;
  }

  function entityRefKey(ref) {
    return [ref?.source_id || "", ref?.ad_id || "", ref?.entity_id || ""].join("\u001f");
  }

  function mergeManualClusters(clusters, manualMatches, kind) {
    const relevant = (manualMatches || []).filter((match) => match.kind === kind);
    if (!relevant.length) return clusters;
    for (const match of relevant) {
      const leftKey = entityRefKey(match.left);
      const rightKey = entityRefKey(match.right);
      let leftIndex = clusters.findIndex((cluster) => cluster.items.some((item) => entityRefKey(item) === leftKey));
      let rightIndex = clusters.findIndex((cluster) => cluster.items.some((item) => entityRefKey(item) === rightKey));
      if (leftIndex < 0 || rightIndex < 0) continue;
      if (leftIndex === rightIndex) continue;
      const left = clusters[leftIndex];
      const right = clusters[rightIndex];
      const sourceIds = new Set(left.items.map((item) => item.source_id));
      if (right.items.some((item) => sourceIds.has(item.source_id))) continue;
      left.items.push(...right.items);
      const keptIndex = Math.min(leftIndex, rightIndex);
      const removedIndex = Math.max(leftIndex, rightIndex);
      clusters.splice(removedIndex, 1);
      if (rightIndex < leftIndex) {
        clusters[keptIndex] = left;
      }
    }
    return clusters.map((cluster, index) => ({
      ...cluster,
      manual_match_count: relevant.filter((match) => {
        const keys = new Set(cluster.items.map(entityRefKey));
        return keys.has(entityRefKey(match.left)) && keys.has(entityRefKey(match.right));
      }).length || cluster.manual_match_count || 0
    }));
  }

  function clusterEntities(records, kind, threshold = 0.25, manualMatches = []) {
    const items = flattenEntities(records, kind).sort((a, b) => {
      const areaA = a.box ? (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) : 0;
      const areaB = b.box ? (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) : 0;
      return areaB - areaA;
    });
    const clusters = [];
    for (const item of items) {
      let best = null;
      let bestIou = 0;
      for (const cluster of clusters) {
        if (cluster.items.some((candidate) => candidate.source_id === item.source_id)) continue;
        const score = Math.max(...cluster.items.map((candidate) => boxIou(candidate.box, item.box)));
        if (score > bestIou) {
          best = cluster;
          bestIou = score;
        }
      }
      if (best && bestIou >= threshold) best.items.push(item);
      else clusters.push({ kind, items: [item] });
    }
    const merged = mergeManualClusters(clusters, manualMatches, kind);
    return merged.map((cluster, index) => ({
      ...cluster,
      cluster_id: `${kind}_${index + 1}`,
      mean_iou: meanPairwiseIou(cluster.items)
    }));
  }

  function meanPairwiseIou(items) {
    const values = [];
    for (let i = 0; i < items.length; i += 1) {
      for (let j = i + 1; j < items.length; j += 1) {
        values.push(boxIou(items[i].box, items[j].box));
      }
    }
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }

  function valueAt(object, path) {
    return path.split(".").reduce((value, key) => value?.[key], object);
  }

  function displayValue(value) {
    if (value === null || value === undefined || value === "" || (Array.isArray(value) && !value.length)) return null;
    if (Array.isArray(value)) return value.join(", ");
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }

  function normalizeComparable(value) {
    const normalized = String(value).trim().toLowerCase();
    const legacyMap = {
      general_crowd: "other_group",
      off_frame: "off_frame_or_scene_direction",
      scene_direction: "off_frame_or_scene_direction",
      "1_not_legible": "0_not_legible",
      "2_low_legibility": "1_low_legibility",
      "3_moderate_legibility": "2_moderate_legibility",
      "4_high_legibility": "3_high_legibility",
      own_body_part: "other_body_part",
      another_person: "part_of_another_person"
    };
    return (legacyMap[normalized] || normalized).replace(/[\s_-]+/g, " ");
  }

  function compareValues(valuesBySource, sourceIds) {
    const present = sourceIds
      .map((sourceId) => displayValue(valuesBySource[sourceId]))
      .filter((value) => value !== null);
    if (present.length < 2) return { state: "insufficient", present: present.length, distinct: present.length };
    const distinct = new Set(present.map(normalizeComparable));
    return { state: distinct.size === 1 ? "agree" : "disagree", present: present.length, distinct: distinct.size };
  }

  function pageRows(records, sourceIds) {
    const definitions = [
      ["Qualifying ads", "page.qualifying_ad_count"],
      ["Page status", "status"]
    ];
    return definitions.map(([label, path]) => rowFromRecords(label, records, sourceIds, path));
  }

  function rowFromRecords(label, records, sourceIds, path) {
    const values = {};
    for (const record of records) values[record.source_id] = valueAt(record.annotation, path);
    return { label, values, comparison: compareValues(values, sourceIds) };
  }

  function clusterRows(cluster, sourceIds, definitions) {
    return definitions.map(([label, path]) => {
      const values = {};
      for (const item of cluster.items) values[item.source_id] = valueAt(item.entity, path);
      return { label, values, comparison: compareValues(values, sourceIds) };
    });
  }

  function summarizeRows(rows) {
    const summary = { agree: 0, disagree: 0, insufficient: 0, comparable: 0, score: null };
    for (const row of rows) {
      summary[row.comparison.state] += 1;
      if (row.comparison.state !== "insufficient") summary.comparable += 1;
    }
    summary.score = summary.comparable ? summary.agree / summary.comparable : null;
    return summary;
  }

  return {
    boxIou,
    clusterEntities,
    clusterRows,
    compareValues,
    displayValue,
    entityRefKey,
    normalizedBox,
    pageRows,
    summarizeRows
  };
});
