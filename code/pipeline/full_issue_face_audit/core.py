from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


REQUIRED_ANNOTATION_COLUMNS = {
    "issue_page_identifier",
    "x1_rel",
    "y1_rel",
    "x2_rel",
    "y2_rel",
    "analyzable",
}
DETECTOR_BOX_COLUMNS = (
    "Bounding Box relative X1",
    "Bounding Box relative Y1",
    "Bounding Box relative X2",
    "Bounding Box relative Y2",
)
ANNOTATION_BOX_COLUMNS = ("x1_rel", "y1_rel", "x2_rel", "y2_rel")
ISSUE_PAGE_RE = re.compile(r"^(\d{4}-\d{4})-(\d{4})$")
DETECTOR_FILENAME_RE = re.compile(r"^(\d{4}-\d{4})-(\d{4}(?:,\d{4})*)_")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collection_sha256(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def parse_box(row: dict[str, str], columns: tuple[str, str, str, str]) -> tuple[float, float, float, float]:
    box = tuple(float(row[column]) for column in columns)
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"invalid normalized box: {box}")
    return box


def parse_detector_box(
    row: dict[str, str],
    columns: tuple[str, str, str, str],
) -> tuple[float, float, float, float]:
    """Preserve reviewed detector coordinates, including small edge overflow."""
    box = tuple(float(row[column]) for column in columns)
    x1, y1, x2, y2 = box
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"non-finite detector box: {box}")
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"invalid detector box ordering: {box}")
    if min(box) < -0.25 or max(box) > 1.25:
        raise ValueError(f"detector box exceeds bounded edge tolerance: {box}")
    return box


def pdf_page_count(pdfinfo: Path, pdf_path: Path) -> int:
    result = subprocess.run(
        [str(pdfinfo), str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise ValueError(f"could not read page count from {pdf_path}")
    return int(match.group(1))


def load_sample_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"issue_id", "decade", "sample_rank"}.issubset(rows[0]):
        raise ValueError("sample manifest lacks required columns")
    by_issue: dict[str, dict[str, Any]] = {}
    for row in rows:
        issue_id = row["issue_id"]
        if issue_id in by_issue:
            raise ValueError(f"duplicate sampled issue: {issue_id}")
        by_issue[issue_id] = {
            "issue_id": issue_id,
            "decade": int(row["decade"]),
            "sample_rank": int(row["sample_rank"]),
            "year": int(row.get("year") or issue_id[5:9]),
        }
    return by_issue, rows


def face_identity(page_id: str, box: tuple[float, float, float, float], analyzable: bool) -> str:
    payload = "|".join(
        [page_id, *(f"{value:.9f}" for value in box), str(analyzable).lower()]
    )
    return f"{page_id}::human_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def load_annotations(
    annotations_dir: Path,
    sample: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(annotations_dir.glob("*_annotations.csv"), key=lambda item: item.name.casefold())
    expected_files = {f"{issue_id}_annotations.csv" for issue_id in sample}
    actual_files = {path.name for path in files}
    if actual_files != expected_files:
        raise ValueError(
            "annotation/sample mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    faces: list[dict[str, Any]] = []
    exact_rows: set[tuple[Any, ...]] = set()
    coordinate_rows: set[tuple[Any, ...]] = set()
    for path in files:
        issue_id = path.name.removesuffix("_annotations.csv")
        short_issue = issue_id.removeprefix("ECON-")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_ANNOTATION_COLUMNS.issubset(reader.fieldnames or []):
                raise ValueError(f"missing annotation columns in {path.name}")
            for source_row, row in enumerate(reader, start=2):
                page_id = row["issue_page_identifier"].strip()
                match = ISSUE_PAGE_RE.fullmatch(page_id)
                if not match or match.group(1) != short_issue:
                    raise ValueError(f"bad page identifier in {path.name}:{source_row}: {page_id}")
                box = parse_box(row, ANNOTATION_BOX_COLUMNS)
                analyzable = parse_bool(row["analyzable"])
                exact = (page_id, *box, analyzable)
                coordinates = (page_id, *box)
                if exact in exact_rows:
                    raise ValueError(f"duplicate annotation row: {exact}")
                if coordinates in coordinate_rows:
                    raise ValueError(f"duplicate annotation coordinates: {coordinates}")
                exact_rows.add(exact)
                coordinate_rows.add(coordinates)
                faces.append({
                    "face_id": face_identity(page_id, box, analyzable),
                    "issue_id": issue_id,
                    "issue_page_identifier": page_id,
                    "pdf_page_index": int(match.group(2)),
                    "bbox_rel": list(box),
                    "bbox_1000": [round(value * 1000) for value in box],
                    "bbox_area_rel": (box[2] - box[0]) * (box[3] - box[1]),
                    "analyzable": analyzable,
                    "decade": sample[issue_id]["decade"],
                    "source_file": path.name,
                    "source_row": source_row,
                })
    if len({face["face_id"] for face in faces}) != len(faces):
        raise ValueError("generated face IDs are not unique")
    return faces, files


def assign_detector_box(
    filename: str,
    box: tuple[float, float, float, float],
) -> tuple[str, tuple[float, float, float, float], str]:
    match = DETECTOR_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"unrecognized detector filename: {filename}")
    issue = match.group(1)
    pages = [int(value) for value in match.group(2).split(",")]
    if len(pages) == 1:
        return f"{issue}-{pages[0]:04d}", box, "single_page"
    if len(pages) != 2:
        raise ValueError(f"audit supports one- and two-page detector scans only: {filename}")
    x1, y1, x2, y2 = box
    center = (x1 + x2) / 2
    if center < 0.5:
        transformed = (x1 * 2, y1, min(1.0, x2 * 2), y2)
        page = pages[0]
        side = "two_page_left"
    else:
        transformed = (max(0.0, x1 * 2 - 1), y1, x2 * 2 - 1, y2)
        page = pages[1]
        side = "two_page_right"
    if not (transformed[0] < transformed[2]):
        raise ValueError(f"invalid transformed spread box for {filename}: {transformed}")
    return f"{issue}-{page:04d}", transformed, side


def load_detector_rows(path: Path, sampled_issues: set[str]) -> list[dict[str, Any]]:
    short_issues = {issue.removeprefix("ECON-") for issue in sampled_issues}
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Filename", *DETECTOR_BOX_COLUMNS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("detector CSV lacks required columns")
        for row_index, row in enumerate(reader, start=2):
            match = DETECTOR_FILENAME_RE.match(row["Filename"])
            if not match or match.group(1) not in short_issues:
                continue
            original_box = parse_detector_box(row, DETECTOR_BOX_COLUMNS)
            page_id, page_box, scan_side = assign_detector_box(row["Filename"], original_box)
            records.append({
                "detector_id": f"detector_row_{row_index:06d}",
                "source_row": row_index,
                "filename": row["Filename"],
                "issue_page_identifier": page_id,
                "bbox_rel": list(page_box),
                "scan_side": scan_side,
            })
    return records


def iou(first: list[float] | tuple[float, ...], second: list[float] | tuple[float, ...]) -> float:
    ix = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    iy = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = ix * iy
    if intersection <= 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def maximum_cardinality_matches(
    faces: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    adjacency: list[list[tuple[float, int]]] = []
    for face in faces:
        edges = [
            (iou(face["bbox_rel"], detection["bbox_rel"]), detection_index)
            for detection_index, detection in enumerate(detections)
        ]
        adjacency.append(sorted((edge for edge in edges if edge[0] >= threshold), reverse=True))
    detection_to_face: dict[int, int] = {}

    def augment(face_index: int, seen: set[int]) -> bool:
        for _, detection_index in adjacency[face_index]:
            if detection_index in seen:
                continue
            seen.add(detection_index)
            if detection_index not in detection_to_face or augment(detection_to_face[detection_index], seen):
                detection_to_face[detection_index] = face_index
                return True
        return False

    order = sorted(
        range(len(faces)),
        key=lambda index: (-max((score for score, _ in adjacency[index]), default=-1), faces[index]["face_id"]),
    )
    for face_index in order:
        augment(face_index, set())
    return [
        (face_index, detection_index, iou(faces[face_index]["bbox_rel"], detections[detection_index]["bbox_rel"]))
        for detection_index, face_index in sorted(detection_to_face.items())
    ]


def match_faces(
    faces: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    faces_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detections_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for face in faces:
        faces_by_page[face["issue_page_identifier"]].append(face)
    for detection in detections:
        detections_by_page[detection["issue_page_identifier"]].append(detection)
    matched_detection_ids: set[str] = set()
    for page_id, page_faces in faces_by_page.items():
        page_detections = detections_by_page.get(page_id, [])
        for face_index, detection_index, score in maximum_cardinality_matches(
            page_faces, page_detections, threshold
        ):
            face = page_faces[face_index]
            detection = page_detections[detection_index]
            face["detection_status"] = "found"
            face["matched_detector_id"] = detection["detector_id"]
            face["match_iou"] = score
            matched_detection_ids.add(detection["detector_id"])
    for face in faces:
        face.setdefault("detection_status", "missed")
        face.setdefault("matched_detector_id", None)
        face.setdefault("match_iou", None)
    for detection in detections:
        detection["match_status"] = (
            "matched_human_face" if detection["detector_id"] in matched_detection_ids else "unmatched"
        )
    return faces


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metric_summary(faces: list[dict[str, Any]], detections: list[dict[str, Any]]) -> dict[str, Any]:
    true_positives = sum(face["detection_status"] == "found" for face in faces)
    false_negatives = len(faces) - true_positives
    false_positives = len(detections) - true_positives
    precision = true_positives / len(detections)
    recall = true_positives / len(faces)
    analyzable = [face for face in faces if face["analyzable"]]
    analyzable_tp = sum(face["detection_status"] == "found" for face in analyzable)
    ignored_predictions = true_positives - analyzable_tp
    evaluated_predictions = len(detections) - ignored_predictions
    analyzable_precision = analyzable_tp / evaluated_predictions
    analyzable_recall = analyzable_tp / len(analyzable)
    return {
        "all_faces": {
            "human_faces": len(faces),
            "dataset_predictions": len(detections),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        },
        "analyzable_faces_ignored_gt_convention": {
            "human_faces": len(analyzable),
            "true_positives": analyzable_tp,
            "false_negatives": len(analyzable) - analyzable_tp,
            "ignored_predictions_matching_non_analyzable_faces": ignored_predictions,
            "evaluated_predictions": evaluated_predictions,
            "false_positives": false_positives,
            "precision": analyzable_precision,
            "recall": analyzable_recall,
            "f1": f1(analyzable_precision, analyzable_recall),
        },
    }


def select_stratified_pilot(
    faces: list[dict[str, Any]],
    size: int = 40,
    seed: str = "full_issue_face_audit_pilot_v1",
) -> list[str]:
    if size <= 0 or size > len(faces):
        raise ValueError("pilot size must be between 1 and the face count")
    groups: dict[tuple[int, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for face in faces:
        groups[(face["decade"], face["detection_status"], face["analyzable"])].append(face)
    for key, rows in groups.items():
        rows.sort(
            key=lambda face: hashlib.sha256(
                f"{seed}|{key}|{face['face_id']}".encode("utf-8")
            ).hexdigest()
        )
    keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest(),
    )
    selected: list[str] = []
    round_index = 0
    while len(selected) < size:
        added = False
        for key in keys:
            rows = groups[key]
            if round_index < len(rows):
                selected.append(rows[round_index]["face_id"])
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
        round_index += 1
    if len(selected) != size:
        raise RuntimeError(f"could select only {len(selected)} pilot faces")
    return selected


def render_pdf_page(
    pdftoppm: Path,
    pdf_path: Path,
    pdf_page_index: int,
    output_path: Path,
    dpi: int,
) -> Path:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_path.with_suffix("")
    page_number = pdf_page_index + 1
    subprocess.run(
        [
            str(pdftoppm),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-jpeg",
            "-r",
            str(dpi),
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        capture_output=True,
    )
    if not output_path.exists():
        raise FileNotFoundError(f"pdftoppm did not create {output_path}")
    with Image.open(output_path) as image:
        if image.width < 500 or image.height < 500:
            raise ValueError(f"unexpectedly small rendered page: {output_path} {image.size}")
    return output_path


def to_pixels(box_1000: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return tuple(
        round(value * (width if index % 2 == 0 else height) / 1000)
        for index, value in enumerate(box_1000)
    )  # type: ignore[return-value]


def expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    width, height = image_size
    return (
        max(0, round(center_x - side / 2)),
        max(0, round(center_y - side / 2)),
        min(width, round(center_x + side / 2)),
        min(height, round(center_y + side / 2)),
    )


def letterbox(image: Image.Image, size: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (size, size + 42), "white")
    work = image.copy()
    work.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel.paste(work, ((size - work.width) // 2, 42 + (size - work.height) // 2))
    ImageDraw.Draw(panel).text((12, 12), label, fill="black", font=ImageFont.load_default())
    return panel


def make_face_composite(page_path: Path, box_1000: list[int]) -> Image.Image:
    with Image.open(page_path) as source:
        page = source.convert("RGB")
    face_pixels = to_pixels(box_1000, page.size)
    tight = page.crop(expand_box(face_pixels, page.size, 1.65))
    local = page.crop(expand_box(face_pixels, page.size, 4.5))
    context = page.copy()
    line_width = max(3, round(min(page.size) / 350))
    ImageDraw.Draw(context).rectangle(face_pixels, outline=(255, 0, 0), width=line_width)
    panels = [
        letterbox(tight, 512, "TARGET FACE - enlarged"),
        letterbox(local, 512, "LOCAL CONTEXT"),
        letterbox(context, 512, "FULL ISSUE PAGE - target in red"),
    ]
    canvas = Image.new("RGB", (1536, 554), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 512, 0))
    return canvas


def save_face_composite(page_path: Path, box_1000: list[int], output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_face_composite(page_path, box_1000).save(
        output_path, "JPEG", quality=92, optimize=True
    )
    return output_path


def count_by(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))
