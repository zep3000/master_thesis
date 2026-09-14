from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from core import (
    collection_sha256,
    file_sha256,
    load_annotations,
    load_detector_rows,
    load_sample_manifest,
    match_faces,
    metric_summary,
    pdf_page_count,
    select_stratified_pilot,
    write_json,
)


HERE = Path(__file__).resolve().parent
EXPECTED_COLLECTION_SHA256 = "b1b79b85353ebf2f98239242ae2a9c8acf8993a87b41a850e3c0bb05ef630425"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit inputs and prepare a forced-box face manifest.")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--detector-csv", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--pdfinfo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "local" / "manifest.json")
    parser.add_argument("--match-iou", type=float, default=0.17)
    parser.add_argument("--pilot-size", type=int, default=40)
    parser.add_argument("--pilot-seed", default="full_issue_face_audit_pilot_v1")
    parser.add_argument("--allow-unreviewed-collection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample, _ = load_sample_manifest(args.sample_manifest.resolve())
    faces, annotation_files = load_annotations(args.annotations_dir.resolve(), sample)
    collection_hash = collection_sha256(annotation_files)
    if collection_hash != EXPECTED_COLLECTION_SHA256 and not args.allow_unreviewed_collection:
        raise ValueError(
            "annotation collection differs from the reviewed authoritative collection: "
            f"{collection_hash} != {EXPECTED_COLLECTION_SHA256}"
        )
    if len(sample) != 35 or len(annotation_files) != 35 or len(faces) != 1599:
        raise ValueError(
            f"unexpected authoritative scope: issues={len(sample)}, files={len(annotation_files)}, "
            f"faces={len(faces)}"
        )
    pdfs = {}
    for issue_id in sorted(sample):
        pdf_path = (args.pdf_dir / f"{issue_id}.pdf").resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"missing sampled issue PDF: {pdf_path}")
        pages = pdf_page_count(args.pdfinfo.resolve(), pdf_path)
        pdfs[issue_id] = {
            "path": str(pdf_path),
            "page_count": pages,
            "size_bytes": pdf_path.stat().st_size,
            "sha256": file_sha256(pdf_path),
        }
    for face in faces:
        page_count = pdfs[face["issue_id"]]["page_count"]
        if not 0 <= face["pdf_page_index"] < page_count:
            raise ValueError(
                f"annotation page out of bounds: {face['face_id']} against {page_count} pages"
            )
    detections = load_detector_rows(args.detector_csv.resolve(), set(sample))
    match_faces(faces, detections, args.match_iou)
    metrics = metric_summary(faces, detections)
    if args.match_iou == 0.17:
        reviewed = metrics["all_faces"]
        if (
            reviewed["true_positives"] != 459
            or reviewed["false_positives"] != 384
            or reviewed["false_negatives"] != 1140
        ):
            raise ValueError(f"cleaned-data match totals differ from reviewed values: {reviewed}")
    pilot_face_ids = select_stratified_pilot(faces, args.pilot_size, args.pilot_seed)
    pilot_set = set(pilot_face_ids)
    pilot_faces = [face for face in faces if face["face_id"] in pilot_set]
    manifest = {
        "schema_version": "full_issue_face_audit_manifest_v1",
        "method": {
            "detector_match_iou": args.match_iou,
            "spread_assignment": "horizontal_center_then_page_relative_x_transform",
            "matching": "maximum_cardinality_one_to_one_within_physical_page",
            "pilot_selection": "deterministic_round_robin_decade_x_status_x_analyzable",
            "pilot_seed": args.pilot_seed,
        },
        "source": {
            "annotations_dir": str(args.annotations_dir.resolve()),
            "annotation_collection_sha256": collection_hash,
            "sample_manifest": str(args.sample_manifest.resolve()),
            "sample_manifest_sha256": file_sha256(args.sample_manifest.resolve()),
            "detector_csv": str(args.detector_csv.resolve()),
            "detector_csv_sha256": file_sha256(args.detector_csv.resolve()),
            "pdf_dir": str(args.pdf_dir.resolve()),
        },
        "audit": {
            "issue_count": len(sample),
            "annotation_file_count": len(annotation_files),
            "annotation_count": len(faces),
            "annotated_page_count": len({face["issue_page_identifier"] for face in faces}),
            "analyzable_count": sum(face["analyzable"] for face in faces),
            "non_analyzable_count": sum(not face["analyzable"] for face in faces),
            "detector_count_in_sampled_issues": len(detections),
            "metrics": metrics,
        },
        "pdfs": pdfs,
        "faces": faces,
        "detections": detections,
        "pilot": {
            "size": len(pilot_face_ids),
            "face_ids": pilot_face_ids,
            "decades": dict(sorted(Counter(str(face["decade"]) for face in pilot_faces).items())),
            "detection_status": dict(
                sorted(Counter(face["detection_status"] for face in pilot_faces).items())
            ),
            "analyzable": dict(
                sorted(Counter(str(face["analyzable"]).lower() for face in pilot_faces).items())
            ),
            "joint_cells": len(
                {
                    (face["decade"], face["detection_status"], face["analyzable"])
                    for face in pilot_faces
                }
            ),
        },
    }
    write_json(args.output.resolve(), manifest)
    print(f"wrote {args.output.resolve()}")
    print(f"annotation collection sha256: {collection_hash}")
    print(f"faces: {len(faces)}; analyzable: {manifest['audit']['analyzable_count']}")
    print(f"detector matches: {metrics['all_faces']['true_positives']}; missed: {metrics['all_faces']['false_negatives']}")
    print(f"pilot: {manifest['pilot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
