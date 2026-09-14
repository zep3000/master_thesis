"""Generate advertisement-detection F1 curves across IoU thresholds.

The figure contrasts two defensible human-reference constructions:

* ``pooled`` micro-pools three independent LLM--human comparisons (A, B, C);
* ``match-any`` collapses human rows with the same image/ad identifier into one
  reference advertisement and accepts the best IoU against any annotator's box.

Only aggregate counts are retained in the figure. The row-level evaluation
tables remain local and outside version control.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "data" / "processed" / "matplotlib"),
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


HUMAN_SOURCES = ("A", "B", "C")
LLM_SOURCE_TYPE = "llm"
THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 100, 5))

BLUE = "#4C78A8"
CORAL = "#E45756"
AMBER = "#F2B701"
TEXT = "#263238"
MUTED = "#66717E"
GRID = "#D9DEE3"
SPINE = "#AAB3BC"


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repository_root / "data" / "processed" / "llm_evaluation_300",
        help="Directory containing annotation_ads.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "code" / "output" / "figures" / "ad-iou-threshold-f1-curves.svg",
        help="SVG output path.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional PNG preview path for visual inspection.",
    )
    return parser.parse_args()


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def best_matching_count(scores: Sequence[Sequence[float]], threshold: float) -> int:
    """Return maximum-cardinality one-to-one matches, breaking ties by IoU."""

    left_count = len(scores)
    right_count = len(scores[0]) if left_count else 0
    if not left_count or not right_count:
        return 0

    # Put the smaller side in the bit mask. Advertisement counts per page are
    # tiny, but this orientation keeps the routine well behaved in audits.
    if right_count > left_count:
        transposed = tuple(tuple(scores[row][column] for row in range(left_count)) for column in range(right_count))
        return best_matching_count(transposed, threshold)

    matrix = tuple(tuple(float(value) for value in row) for row in scores)

    @lru_cache(maxsize=None)
    def solve(left_index: int, used_right: int) -> tuple[int, float]:
        if left_index == left_count:
            return 0, 0.0
        best = solve(left_index + 1, used_right)
        for right_index, score in enumerate(matrix[left_index]):
            if used_right & (1 << right_index) or score < threshold:
                continue
            subsequent_count, subsequent_score = solve(left_index + 1, used_right | (1 << right_index))
            candidate = subsequent_count + 1, subsequent_score + score
            if candidate > best:
                best = candidate
        return best

    return solve(0, 0)[0]


def boxes(frame: pd.DataFrame) -> list[tuple[float, float, float, float]]:
    columns = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    return [tuple(map(float, row)) for row in frame[columns].itertuples(index=False, name=None)]


def f1_score(gold_total: int, predicted_total: int, matched: int) -> float:
    denominator = gold_total + predicted_total
    return 2 * matched / denominator if denominator else 0.0


def pair_scores(
    left: Iterable[Sequence[float]],
    right: Iterable[Sequence[float]],
) -> list[list[float]]:
    right_boxes = list(right)
    return [[iou(left_box, right_box) for right_box in right_boxes] for left_box in left]


def pooled_curve(human: pd.DataFrame, predicted: pd.DataFrame) -> list[float]:
    gold_total = len(human)
    predicted_total = len(predicted) * len(HUMAN_SOURCES)
    grouped_predicted = {image_id: boxes(frame) for image_id, frame in predicted.groupby("image_id", sort=False)}
    comparisons: list[list[list[float]]] = []
    for source in HUMAN_SOURCES:
        source_rows = human[human["source_id"] == source]
        for image_id, frame in source_rows.groupby("image_id", sort=False):
            comparisons.append(pair_scores(boxes(frame), grouped_predicted.get(image_id, [])))

    return [
        f1_score(gold_total, predicted_total, sum(best_matching_count(scores, threshold) for scores in comparisons))
        for threshold in THRESHOLDS
    ]


def match_any_curve(human: pd.DataFrame, predicted: pd.DataFrame) -> list[float]:
    # ad_id is assigned within each page by the annotation interface. Grouping
    # it across annotators creates a stable reference entity whose spatial score
    # is the maximum IoU with any available human box.
    references: dict[str, list[list[tuple[float, float, float, float]]]] = {}
    for image_id, image_rows in human.groupby("image_id", sort=False):
        references[str(image_id)] = [
            boxes(frame)
            for _, frame in image_rows.groupby("ad_id", sort=False)
        ]

    grouped_predicted = {str(image_id): boxes(frame) for image_id, frame in predicted.groupby("image_id", sort=False)}
    page_scores: list[list[list[float]]] = []
    for image_id, image_references in references.items():
        predicted_boxes = grouped_predicted.get(image_id, [])
        page_scores.append(
            [
                [max((iou(human_box, predicted_box) for human_box in reference), default=0.0) for predicted_box in predicted_boxes]
                for reference in image_references
            ]
        )

    reference_total = sum(len(image_references) for image_references in references.values())
    predicted_total = len(predicted)
    return [
        f1_score(reference_total, predicted_total, sum(best_matching_count(scores, threshold) for scores in page_scores))
        for threshold in THRESHOLDS
    ]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.hashsalt": "ad-iou-threshold-f1-curves-v1",
        }
    )


def draw_figure(pooled: Sequence[float], match_any: Sequence[float], output: Path, preview: Path | None) -> None:
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(9.4, 4.9), facecolor="white")
    figure.subplots_adjust(left=0.095, right=0.975, bottom=0.19, top=0.76)

    axis.plot(
        THRESHOLDS,
        pooled,
        color=BLUE,
        linewidth=2.25,
        marker="o",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.3,
        label="Pooled human comparisons",
        zorder=3,
    )
    axis.plot(
        THRESHOLDS,
        match_any,
        color=CORAL,
        linewidth=2.25,
        linestyle="--",
        marker="D",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Match-any human reference",
        zorder=3,
    )

    axis.axvline(0.20, color=AMBER, linewidth=1.4, linestyle=(0, (3, 3)), zorder=1)
    axis.axvline(0.50, color=SPINE, linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)
    axis.text(0.20, 0.045, "IoU 0.20", color="#8A6900", ha="center", va="bottom", fontsize=8.5)
    axis.text(0.50, 0.045, "IoU 0.50", color=MUTED, ha="center", va="bottom", fontsize=8.5)

    axis.set_xlim(0.04, 0.96)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0.05, 0.20, 0.50, 0.70, 0.90])
    axis.set_xticklabels(["0.05", "0.20", "0.50", "0.70", "0.90"])
    axis.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    axis.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    axis.set_xlabel("Advertisement IoU threshold", fontsize=10.5, fontweight="semibold", labelpad=10)
    axis.set_ylabel("Advertisement detection F1", fontsize=10.5, fontweight="semibold", labelpad=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", length=0, labelsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE)
        axis.spines[side].set_linewidth(0.8)

    figure.text(0.095, 0.925, "Advertisement detection across IoU thresholds", fontsize=15.5, fontweight="semibold", ha="left")
    figure.text(
        0.095,
        0.865,
        "LLM evaluation against three human annotators on the 300-page evaluation set",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
        frameon=False,
        fontsize=9.3,
        handlelength=3.2,
        columnspacing=2.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", facecolor="white", metadata={"Date": None})
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=180, facecolor="white", metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ads_path = args.input_dir / "annotation_ads.parquet"
    if not ads_path.is_file():
        raise FileNotFoundError(f"Evaluation table not found: {ads_path}")

    ads = pd.read_parquet(ads_path)
    required = {"source_id", "source_type", "image_id", "ad_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    missing = required.difference(ads.columns)
    if missing:
        raise ValueError(f"annotation_ads.parquet lacks required columns: {sorted(missing)}")
    if ads[list(required - {"source_id", "source_type", "image_id", "ad_id"})].isna().any().any():
        raise ValueError("Advertisement bounding-box coordinates contain missing values")

    human = ads[ads["source_id"].isin(HUMAN_SOURCES)].copy()
    predicted = ads[ads["source_type"] == LLM_SOURCE_TYPE].copy()
    found_sources = set(human["source_id"].unique())
    if found_sources != set(HUMAN_SOURCES):
        raise ValueError(f"Expected human sources {HUMAN_SOURCES}, found {sorted(found_sources)}")
    if predicted.empty:
        raise ValueError("No LLM advertisement rows found")

    pooled = pooled_curve(human, predicted)
    match_any = match_any_curve(human, predicted)
    draw_figure(pooled, match_any, args.output, args.preview)

    print("threshold\tpooled_humans\tmatch_any_human")
    for threshold, pooled_value, match_any_value in zip(THRESHOLDS, pooled, match_any):
        print(f"{threshold:.2f}\t{pooled_value:.6f}\t{match_any_value:.6f}")


if __name__ == "__main__":
    main()
