"""Export small, publication-ready DataFrames as native Quarto tables.

Notebooks in ``code/scripts`` can import the public function directly::

    from thesis_tables import export_quarto_table

The exporter deliberately writes two artifacts:

* a CSV containing the selected, unformatted values; and
* a generated QMD fragment containing the formatted Pandoc pipe table.

This keeps the analytical values inspectable while allowing Quarto to typeset
the final table natively in PDF, DOCX, and HTML output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

import pandas as pd


Formatter = str | Callable[[Any], object]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "code" / "output" / "tables"
DEFAULT_DATA_DIR = DEFAULT_ARTIFACT_DIR
DEFAULT_QMD_DIR = DEFAULT_ARTIFACT_DIR

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SAFE_LABEL = re.compile(r"^tbl-[a-z0-9][a-z0-9_-]*$")
_ALIGNMENT_MARKERS = {
    "left": ":---",
    "center": ":---:",
    "right": "---:",
}


def _is_missing(value: Any) -> bool:
    """Return whether *value* is a scalar missing value."""

    result = pd.isna(value)
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _format_value(
    value: Any,
    formatter: Formatter | None,
    *,
    missing: str,
    float_precision: int,
) -> str:
    if _is_missing(value):
        return missing

    if formatter is not None:
        if callable(formatter):
            return str(formatter(value))
        if "{" in formatter:
            return formatter.format(value)
        return format(value, formatter)

    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Integral):
        return f"{value:,d}"
    if isinstance(value, Real):
        return f"{value:,.{float_precision}f}"
    return str(value)


def _escape_cell(value: str) -> str:
    """Keep a value within one Pandoc pipe-table cell."""

    return " ".join(value.splitlines()).replace("|", r"\|").strip()


def _default_alignment(series: pd.Series) -> str:
    return "right" if pd.api.types.is_numeric_dtype(series.dtype) else "left"


def export_quarto_table(
    table: pd.DataFrame,
    name: str,
    *,
    caption: str,
    label: str,
    column_labels: Mapping[str, str] | None = None,
    formats: Mapping[str, Formatter] | None = None,
    alignments: Mapping[str, str] | None = None,
    note: str | None = None,
    include_index: bool = False,
    missing: str = "—",
    float_precision: int = 2,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    qmd_dir: str | Path = DEFAULT_QMD_DIR,
) -> tuple[Path, Path]:
    """Write raw table values and a formatted Quarto table fragment.

    Parameters
    ----------
    table:
        A small DataFrame already reduced to the rows and columns intended for
        the thesis. Multi-level columns are intentionally unsupported.
    name:
        Stable lowercase artifact name, without an extension.
    caption, label:
        The visible caption and Quarto cross-reference label. ``label`` must
        begin with ``tbl-``.
    column_labels:
        Optional display names keyed by DataFrame column name.
    formats:
        Optional Python format specifications or callables keyed by column.
        Examples: ``{"share": ".1%", "kappa": ".2f", "n": ",d"}``.
    alignments:
        Optional ``left``, ``center``, or ``right`` overrides keyed by column.
        Numeric columns default to right alignment; other columns default left.
    note:
        Optional Markdown note placed directly beneath the caption.
    include_index:
        Include the DataFrame index as the first exported column.
    missing:
        Display string for missing values in the QMD fragment.
    float_precision:
        Default decimals for floating-point columns without an explicit format.
    data_dir, qmd_dir:
        Override output directories, primarily for tests or exceptional uses.

    Returns
    -------
    tuple[Path, Path]
        Paths to the CSV values and QMD presentation fragment, respectively.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if table.empty:
        raise ValueError("refusing to export an empty thesis table")
    if isinstance(table.columns, pd.MultiIndex):
        raise ValueError("flatten multi-level columns before exporting")
    if not table.columns.is_unique:
        raise ValueError("table columns must be unique")
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError("name must contain only lowercase letters, digits, '-' or '_'")
    if not caption.strip():
        raise ValueError("caption must not be empty")
    if not _SAFE_LABEL.fullmatch(label):
        raise ValueError("label must begin with 'tbl-' and use lowercase safe characters")
    if float_precision < 0:
        raise ValueError("float_precision must be non-negative")

    exported = table.reset_index() if include_index else table.copy()
    columns = list(exported.columns)
    column_set = set(columns)
    column_labels = dict(column_labels or {})
    formats = dict(formats or {})
    alignments = dict(alignments or {})

    for option_name, values in (
        ("column_labels", column_labels),
        ("formats", formats),
        ("alignments", alignments),
    ):
        unknown = set(values) - column_set
        if unknown:
            raise KeyError(f"{option_name} contains unknown columns: {sorted(unknown)!r}")

    invalid_alignments = {
        column: alignment
        for column, alignment in alignments.items()
        if alignment not in _ALIGNMENT_MARKERS
    }
    if invalid_alignments:
        raise ValueError(
            "alignments must be 'left', 'center', or 'right': "
            f"{invalid_alignments!r}"
        )

    headers = [_escape_cell(str(column_labels.get(column, column))) for column in columns]
    resolved_alignments = [
        alignments.get(column, _default_alignment(exported[column])) for column in columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(_ALIGNMENT_MARKERS[value] for value in resolved_alignments) + "|",
    ]

    for row in exported.itertuples(index=False, name=None):
        cells = [
            _escape_cell(
                _format_value(
                    value,
                    formats.get(column),
                    missing=missing,
                    float_precision=float_precision,
                )
            )
            for column, value in zip(columns, row, strict=True)
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(("", f": {caption.strip()} {{#{label}}}"))
    if note and note.strip():
        lines.extend(("", f"*Note.* {note.strip()}"))
    lines.append("")

    data_path = Path(data_dir) / f"{name}.csv"
    qmd_path = Path(qmd_dir) / f"{name}.qmd"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    qmd_path.parent.mkdir(parents=True, exist_ok=True)

    exported.to_csv(data_path, index=False)
    qmd_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return data_path, qmd_path


__all__ = ["export_quarto_table"]
