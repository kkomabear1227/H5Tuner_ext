#!/usr/bin/env python3
"""Grouped-bar characterization figure for the h5bench exerciser sweep on Nurion.

Two panels, each answering one message:
  (a) MPI-IO collectivity  -> the winning mode flips with array shape and scale
  (b) Lustre stripe layout -> the best layout differs per context

Bars are normalized to the best configuration inside each group, so the bar
heights read as "fraction of the achievable bandwidth in this context".  The
absolute peak and the max/min spread are carried in the table below the axis.
Whiskers are the min-max over the exerciser's 10 internal iterations.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).parent
OUT  = Path.home() / "Downloads" / "mpiio-parameter-sensitivity.png"

# ---- design tokens (validated: see dataviz validate_palette.js) ------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#84837c"
RULE      = "#d9d8d3"
GRID      = "#e8e7e3"
CAT       = ["#2a78d6", "#eb6834"]                 # categorical slots 1-2
ORD       = ["#86b6ef", "#2a78d6", "#104281"]      # ordinal blue 250/450/650

DIMS_LABEL = {1: "1D array", 2: "2D array", 3: "3D array"}
STRIPES    = ["1M-c1", "4M-c4", "64M-c16"]
STRIPE_TXT = {"1M-c1": "1 MiB × 1", "4M-c4": "4 MiB × 4",
              "64M-c16": "64 MiB × 16"}


def load():
    rows = []
    with open(HERE / "exerciser.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "dims": int(r["dims"]), "io_mode": r["io_mode"],
                "np": int(r["np"]), "stripe": r["stripe"],
                "avg": float(r["bw_avg_GBs"]),
                "lo": float(r["bw_min_GBs"]), "hi": float(r["bw_max_GBs"]),
            })
    return rows


def pick(rows, **kw):
    for r in rows:
        if all(r[k] == v for k, v in kw.items()):
            return r
    raise KeyError(kw)


# ---------------------------------------------------------------------------
# panel machinery
# ---------------------------------------------------------------------------
BAR_W   = 0.26          # bar width in group-units
GAP     = 0.02          # surface gap between adjacent bars
TABLE_H = 0.42          # table height in axis data-units (below y=0)


def draw_panel(ax, groups, series, colors, series_names, row_specs,
               block_spans, title, subtitle, table_h=0.42, bar_w=BAR_W):
    """groups: list of dicts with 'cells' (list of {avg,lo,hi}) per series."""
    n_groups = len(groups)
    n_series = len(series)

    # -- geometry ----------------------------------------------------------
    total = n_series * bar_w + (n_series - 1) * GAP
    offs = [-total / 2 + bar_w / 2 + i * (bar_w + GAP) for i in range(n_series)]

    n_rows = len(row_specs)
    row_h = table_h / (n_rows + 1)          # +1 for the block-label row
    y_bot = -table_h

    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(y_bot, 1.0)

    # -- gridlines (recessive, plot region only) ---------------------------
    for yv in (0.25, 0.50, 0.75, 1.00):
        ax.plot([-0.5, n_groups - 0.5], [yv, yv], color=GRID, lw=0.7,
                zorder=0, solid_capstyle="butt")

    # -- bars --------------------------------------------------------------
    for gi, g in enumerate(groups):
        best = max(c["avg"] for c in g["cells"])
        for si, cell in enumerate(g["cells"]):
            x = gi + offs[si]
            h = cell["avg"] / best
            ax.add_patch(Rectangle(
                (x - bar_w / 2, 0), bar_w, h,
                facecolor=colors[si], edgecolor=SURFACE, lw=0.8, zorder=3))
            # min-max whisker over the 10 exerciser iterations
            lo, hi = cell["lo"] / best, cell["hi"] / best
            ax.plot([x, x], [lo, hi], color=INK, lw=0.8, zorder=4,
                    solid_capstyle="butt", alpha=0.55)
            ax.plot([x - bar_w * 0.22, x + bar_w * 0.22], [hi, hi],
                    color=INK, lw=0.8, zorder=4, alpha=0.55)
            ax.plot([x - bar_w * 0.22, x + bar_w * 0.22], [lo, lo],
                    color=INK, lw=0.8, zorder=4, alpha=0.55)

    # -- plot-region frame + zero rule -------------------------------------
    ax.plot([-0.5, n_groups - 0.5], [0, 0], color=INK, lw=0.9, zorder=5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # left edge of the plot region only
    ax.plot([-0.5, -0.5], [0, 1.0], color=INK, lw=0.9, zorder=5)
    ax.plot([n_groups - 0.5, n_groups - 0.5], [0, 1.0], color=RULE, lw=0.7, zorder=5)
    ax.plot([-0.5, n_groups - 0.5], [1.0, 1.0], color=RULE, lw=0.7, zorder=5)

    # -- y axis ------------------------------------------------------------
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=8, color=INK_2)
    ax.tick_params(axis="y", length=2.5, width=0.7, color=RULE, pad=2)
    ax.set_xticks([])

    # -- annotation table --------------------------------------------------
    # horizontal rules
    for k in range(n_rows + 2):
        y = -k * row_h
        ax.plot([-0.5, n_groups - 0.5], [y, y], color=RULE, lw=0.7,
                zorder=5, clip_on=False)
    # vertical rules: one per group boundary inside the table
    for gi in range(n_groups + 1):
        x = gi - 0.5
        ax.plot([x, x], [0, -n_rows * row_h], color=RULE, lw=0.7,
                zorder=5, clip_on=False)

    # data rows
    lab_tr = blended_transform_factory(ax.transAxes, ax.transData)
    for ri, spec in enumerate(row_specs):
        yc = -(ri + 0.5) * row_h
        ax.text(-0.012, yc, spec["label"], ha="right", va="center",
                fontsize=7.5, color=INK_2, clip_on=False, transform=lab_tr)
        for gi, val in enumerate(spec["values"]):
            ax.text(gi, yc, val, ha="center", va="center",
                    fontsize=spec.get("size", 7.5),
                    color=spec.get("color", INK),
                    fontweight=spec.get("weight", "normal"), clip_on=False)

    # block-label row (spans several groups)
    y_block = -(n_rows + 0.5) * row_h
    for (lo_g, hi_g, label) in block_spans:
        ax.text((lo_g + hi_g) / 2, y_block, label, ha="center", va="center",
                fontsize=8.5, color=INK, fontweight="bold", clip_on=False)
        # separators between blocks, drawn through the whole panel
        if hi_g < n_groups - 1:
            xb = hi_g + 0.5
            ax.plot([xb, xb], [1.0, -(n_rows + 1) * row_h], color=INK_2,
                    lw=0.8, zorder=6, clip_on=False)
    ax.plot([-0.5, n_groups - 0.5], [-(n_rows + 1) * row_h] * 2,
            color=INK, lw=0.9, zorder=5, clip_on=False)
    ax.plot([-0.5, -0.5], [-n_rows * row_h, -(n_rows + 1) * row_h],
            color=INK, lw=0.9, zorder=5, clip_on=False)
    ax.plot([n_groups - 0.5, n_groups - 0.5],
            [-n_rows * row_h, -(n_rows + 1) * row_h],
            color=INK, lw=0.9, zorder=5, clip_on=False)

    # -- titles ------------------------------------------------------------
    ax.text(0.0, 1.31, title, ha="left", va="baseline", fontsize=10.5,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.0, 1.20, subtitle, ha="left", va="baseline", fontsize=8.5,
            color=INK_2, transform=ax.transAxes)

    # -- legend ------------------------------------------------------------
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=SURFACE, lw=0.8)
               for c in colors]
    handles.append(Line2D([0], [0], color=INK, lw=0.8, alpha=0.55))
    labels = list(series_names) + ["min–max of 10 iterations"]
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.025), ncol=len(labels), frameon=True,
              framealpha=1.0, edgecolor=RULE, facecolor=SURFACE,
              fontsize=8, handlelength=1.3, handleheight=0.85,
              columnspacing=1.1, handletextpad=0.5, borderpad=0.45)


def main():
    rows = load()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(11.6, 7.6))
    gs = fig.add_gridspec(2, 1, hspace=0.50, left=0.175, right=0.985,
                          top=0.880, bottom=0.105)

    # ================= panel (a): collectivity, stripe fixed ===============
    ax_a = fig.add_subplot(gs[0])
    groups_a, best_a, spread_a, np_a = [], [], [], []
    for d in (1, 2, 3):
        for npv in (8, 64, 256):
            cells = [pick(rows, dims=d, io_mode=m, np=npv, stripe="4M-c4")
                     for m in ("collective", "independent")]
            groups_a.append({"cells": cells})
            vals = [c["avg"] for c in cells]
            best_a.append(f"{max(vals):.2f}")
            spread_a.append(f"{max(vals)/min(vals):.1f}×")
            np_a.append(str(npv))

    draw_panel(
        ax_a, groups_a,
        series=["collective", "independent"], colors=CAT,
        series_names=["Collective I/O", "Independent I/O"],
        row_specs=[
            {"label": "Spread (max/min)", "values": spread_a,
             "weight": "bold", "color": INK},
            {"label": "Best [GB/s]", "values": best_a, "color": INK_2},
            {"label": "MPI ranks", "values": np_a, "color": INK},
        ],
        block_spans=[(0, 2, "1D array"), (3, 5, "2D array"), (6, 8, "3D array")],
        title="(a)  MPI-IO collectivity — the winning mode flips with array shape and scale",
        subtitle="Lustre stripe held at 4 MiB × 4.  Independent I/O wins 4 of the 9 contexts "
                 "and loses by 5.8× in another.",
        table_h=0.42, bar_w=0.26,
    )
    ax_a.set_ylabel("Normalized write bandwidth\n(best in group = 1)",
                    fontsize=8.5, color=INK_2)
    ax_a.yaxis.set_label_coords(-0.128, 0.5)

    # ================= panel (b): stripe layout, np fixed ==================
    ax_b = fig.add_subplot(gs[1])
    groups_b, best_b, spread_b, mode_b, win_b = [], [], [], [], []
    for d in (1, 2, 3):
        for m in ("collective", "independent"):
            cells = [pick(rows, dims=d, io_mode=m, np=256, stripe=s)
                     for s in STRIPES]
            groups_b.append({"cells": cells})
            vals = [c["avg"] for c in cells]
            best_b.append(f"{max(vals):.2f}")
            spread_b.append(f"{max(vals)/min(vals):.1f}×")
            winner = STRIPES[vals.index(max(vals))]
            mode_b.append(m)
            win_b.append(STRIPE_TXT[winner])

    draw_panel(
        ax_b, groups_b,
        series=STRIPES, colors=ORD,
        series_names=[STRIPE_TXT[s] for s in STRIPES],
        row_specs=[
            {"label": "Spread (max/min)", "values": spread_b,
             "weight": "bold", "color": INK},
            {"label": "Best [GB/s]", "values": best_b, "color": INK_2},
            {"label": "Best stripe layout", "values": win_b, "size": 7.5,
             "weight": "bold", "color": INK},
            {"label": "I/O mode", "values": mode_b, "color": INK},
        ],
        block_spans=[(0, 1, "1D array"), (2, 3, "2D array"), (4, 5, "3D array")],
        title="(b)  Lustre stripe layout — no single setting is best; the optimum moves with the context",
        subtitle="256 MPI ranks throughout.  Three different stripe layouts take first place "
                 "across the six contexts.",
        table_h=0.56, bar_w=0.24,
    )
    ax_b.set_ylabel("Normalized write bandwidth\n(best in group = 1)",
                    fontsize=8.5, color=INK_2)
    ax_b.yaxis.set_label_coords(-0.128, 0.5)

    fig.text(0.175, 0.038,
             "h5bench exerciser on Nurion (KISTI) · 30 PBS jobs, 2026-03-30 · "
             "128 MiB per rank × 10 iterations · GCC 10.2 + OpenMPI 3.1.0",
             fontsize=7.5, color=INK_MUTED, ha="left")
    fig.text(0.175, 0.014,
             "Bar height is the benchmark's own aggregate write bandwidth "
             "(slowest rank), averaged over the 10 iterations.",
             fontsize=7.5, color=INK_MUTED, ha="left")

    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)
    pdf = OUT.with_suffix(".pdf")
    fig.savefig(pdf)                       # vector, for LaTeX
    print("wrote", pdf)


if __name__ == "__main__":
    main()
