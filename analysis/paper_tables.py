"""The paper's tables: one dataframe each, written as CSV, a LaTeX tabular fragment and Markdown.

`write_table` writes, from the same dataframe and column spec:

    {name}.csv                 every column at full precision, with n per statistic and the
                               trial set the rows use
    {name}.tex                 a plain tabular fragment (\\hline only, no packages): mean +/- sd
                               as $0.6717 \\pm 0.0044$, 4 decimals for accuracy, 3 for HD values,
                               1 for RTA
    {name}.md                  a Markdown preview of the same cells
    {name}_caption_notes.txt   what a caption needs: n, trial set, pending cells, post-hoc
                               rows; never inside the tabular

A cell that waits for runs still training reads "pending"; a cell that does not apply reads
"--" (Markdown "—"). Section rows (the dataset, the grid) span the table. A reference row
(the centralized baseline) has no algorithm, so its value spans the algorithm columns.

`estimate_width` measures a fragment with the font metrics of Times New Roman at the given
size, with \\tabcolsep on both sides of each column; no LaTeX is installed here, so this is an
estimate for deciding the layout, not a typeset width.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.recording import atomic_write_text

ACC, HD, RTA = 4, 3, 1  # decimals
PENDING = "pending"


@dataclass(frozen=True)
class Col:
    """One displayed column.

    kind: "text" (source column as is), "num" (one number), "pm" (mean, sd), "signs"
    (positive, negative counts), "count" (k of n). `ref` gives the source columns a
    reference row uses instead, and `ref_group` merges consecutive columns for it.
    """

    key: str
    header: str
    kind: str
    source: tuple[str, ...]
    decimals: int = ACC
    align: str = "r"
    ref: tuple[str, ...] | None = None
    ref_group: str | None = None


@dataclass
class Table:
    name: str
    df: pd.DataFrame  # one row per displayed row; `section`, `row_kind`, `pending` columns
    columns: list[Col]
    notes: list[str] = field(default_factory=list)
    tabcolsep_pt: float | None = None
    trial_set: str = ""


# ---------------------------------------------------------------- cell formatting
def _isnan(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def _fixed(v, decimals: int) -> str:
    """v at `decimals` places, with no sign on a value that rounds to zero."""
    s = f"{v:.{decimals}f}"
    return s.lstrip("-") if float(s) == 0 else s


def fmt_num(v, decimals: int, latex: bool) -> str:
    if _isnan(v):
        return "--" if latex else "—"
    s = _fixed(v, decimals)
    if latex:
        return f"${s}$"
    return s.replace("-", "−")


def fmt_pm(mean, sd, decimals: int, latex: bool) -> str:
    if _isnan(mean):
        return "--" if latex else "—"
    m = _fixed(mean, decimals)
    if _isnan(sd):
        return f"${m}$" if latex else m.replace("-", "−")
    s = _fixed(sd, decimals)
    if latex:
        return f"${m} \\pm {s}$"
    return f"{m} ± {s}".replace("-", "−")


def fmt_signs(pos, neg, latex: bool, zero=0) -> str:
    """Trials with a positive and a negative difference, and the ties (=) when there are any."""
    if _isnan(pos):
        return "--" if latex else "—"
    tie = 0 if _isnan(zero) else int(zero)
    if latex:
        return f"${int(pos)}{{+}}/{int(neg)}{{-}}" + (f"/{tie}{{=}}" if tie else "") + "$"
    return f"{int(pos)}+/{int(neg)}−" + (f"/{tie}=" if tie else "")


def fmt_count(k, n, latex: bool) -> str:
    if _isnan(k):
        return "--" if latex else "—"
    return f"{int(k)}/{int(n)}"


_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
    "−": "$-$",
    "×": r"$\times$",
    "±": r"$\pm$",
    "μ": r"$\mu$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "Δ": r"$\Delta$",
    "²": r"$^2$",
    "—": "---",
    "–": "--",
}


def latex_text(s) -> str:
    """Plain text for a tabular cell: LaTeX specials escaped, the few non-ASCII symbols as math."""
    if _isnan(s):
        return "--"
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(s))


def render_cell(row: pd.Series, col: Col, latex: bool, reference: bool = False) -> str:
    pending = str(row.get("pending") or "").split(";")
    if col.key in pending or (reference and col.ref_group in pending):
        return PENDING
    source = col.ref if reference and col.ref else col.source
    values = [row.get(s) for s in source]
    if col.kind == "text" or isinstance(values[0], str):  # a reference label in a number column
        v = values[0]
        return latex_text(v) if latex else ("—" if _isnan(v) else str(v))
    if col.kind == "num":
        return fmt_num(values[0], col.decimals, latex)
    if col.kind == "pm":
        return fmt_pm(values[0], values[1], col.decimals, latex)
    if col.kind == "signs":
        return fmt_signs(values[0], values[1], latex, values[2] if len(values) > 2 else 0)
    if col.kind == "count":
        return fmt_count(values[0], values[1], latex)
    raise ValueError(f"unknown column kind {col.kind}")


def _row_cells(row: pd.Series, columns: list[Col], latex: bool) -> list[tuple[str, int, str]]:
    """(text, span, align) per cell; a reference row merges each run of columns sharing a
    ref_group (left-aligned if its first column is, else centred)."""
    reference = row.get("row_kind") == "reference"
    cells, i = [], 0
    while i < len(columns):
        col = columns[i]
        if reference and col.ref_group:
            j = i
            while j + 1 < len(columns) and columns[j + 1].ref_group == col.ref_group:
                j += 1
            align = "l" if col.align == "l" else "c"
            cells.append((render_cell(row, col, latex, reference=True), j - i + 1, align))
            i = j + 1
            continue
        cells.append((render_cell(row, col, latex, reference=reference), 1, col.align))
        i += 1
    return cells


def latex_header(header: str, align: str) -> str:
    """A header; one with line breaks ("\\n") becomes a nested tabular (core LaTeX)."""
    lines = header.split("\n")
    if len(lines) == 1:
        return latex_text(header)
    a = "l" if align == "l" else "r"
    inner = r" \\ ".join(latex_text(x) for x in lines)
    return f"\\begin{{tabular}}[b]{{@{{}}{a}@{{}}}}{inner}\\end{{tabular}}"


# ---------------------------------------------------------------- writers
def to_latex(table: Table) -> str:
    cols = table.columns
    spec = "".join("l" if c.align == "l" else "r" for c in cols)
    lines = []
    if table.tabcolsep_pt is not None:
        lines.append(f"{{\\setlength{{\\tabcolsep}}{{{table.tabcolsep_pt:g}pt}}%")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(latex_header(c.header, c.align) for c in cols) + " \\\\")
    lines.append("\\hline")
    section = None
    for _, row in table.df.iterrows():
        if "section" in row and row["section"] != section:
            section = row["section"]
            lines.append(
                f"\\multicolumn{{{len(cols)}}}{{l}}{{\\textit{{{latex_text(section)}}}}} \\\\"
            )
        parts = []
        for text, span, align in _row_cells(row, cols, latex=True):
            parts.append(text if span == 1 else f"\\multicolumn{{{span}}}{{{align}}}{{{text}}}")
        lines.append(" & ".join(parts) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if table.tabcolsep_pt is not None:
        lines.append("}")
    return "\n".join(lines) + "\n"


def to_markdown(table: Table) -> str:
    cols = table.columns
    lines = [
        "| " + " | ".join(c.header.replace("\n", " ") for c in cols) + " |",
        "|" + "|".join(":--" if c.align == "l" else "--:" for c in cols) + "|",
    ]
    section = None
    for _, row in table.df.iterrows():
        if "section" in row and row["section"] != section:
            section = row["section"]
            lines.append(f"| **{section}** |" + " |" * (len(cols) - 1))
        cells = []
        for text, span, _ in _row_cells(row, cols, latex=False):
            cells += [text] + [""] * (span - 1)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table(table: Table, directory: Path) -> list[Path]:
    directory = Path(directory)
    csv = table.df.copy()
    csv.insert(0, "trial_set", table.trial_set)
    paths = {
        "csv": directory / f"{table.name}.csv",
        "tex": directory / f"{table.name}.tex",
        "md": directory / f"{table.name}.md",
        "notes": directory / f"{table.name}_caption_notes.txt",
    }
    atomic_write_text(paths["csv"], csv.to_csv(index=False, lineterminator="\n"))
    atomic_write_text(paths["tex"], to_latex(table))
    notes = "\n".join(f"- {n}" for n in table.notes)
    atomic_write_text(paths["md"], f"# {table.name}\n\n{to_markdown(table)}\n{notes}\n")
    atomic_write_text(paths["notes"], notes + "\n")
    return list(paths.values())


# ---------------------------------------------------------------- width estimate
def _text_width_in(text: str, size_pt: float) -> float:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    text = text.replace("{+}", "+").replace("{-}", "-").replace("\\textit", "")
    text = re.sub(r"\\multicolumn\{\d+\}\{.\}\{(.*)\}", r"\1", text)
    if not text.strip():
        return 0.0
    try:
        path = TextPath((0, 0), text, prop=FontProperties(family="Times New Roman", size=size_pt))
    except ValueError:
        path = TextPath(
            (0, 0),
            text.replace("$", ""),
            prop=FontProperties(family="Times New Roman", size=size_pt),
        )
    return path.get_extents().width / 72.0


def estimate_width(table: Table, size_pt: float = 8.0) -> float:
    """Width in inches of the widest cell per column, plus 2 x \\tabcolsep per column."""
    sep = 6.0 if table.tabcolsep_pt is None else table.tabcolsep_pt
    widths = [
        max(_text_width_in(latex_text(line), size_pt) for line in c.header.split("\n"))
        for c in table.columns
    ]
    spans = []
    for _, row in table.df.iterrows():
        position = 0
        for text, span, _ in _row_cells(row, table.columns, latex=True):
            if span == 1:
                widths[position] = max(widths[position], _text_width_in(text, size_pt))
            else:
                spans.append((position, span, _text_width_in(text, size_pt)))
            position += span
    for position, span, width in spans:  # a merged cell wider than its columns widens the last
        inner = sum(widths[position : position + span]) + (span - 1) * 2 * sep / 72.0
        if width > inner:
            widths[position + span - 1] += width - inner
    return float(sum(widths) + len(widths) * 2 * sep / 72.0)
