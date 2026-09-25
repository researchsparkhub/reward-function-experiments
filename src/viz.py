"""Rendering helpers: (a) small PNG frames of the grid, used both as the
literal "starting / ending images" fed to the LLM judge and as report
figures, and (b) the comparison-overlay plots used in the Results section.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gridworld as g

COLORS = {
    "wall": "#1f2937",
    "free": "#f8fafc",
    "grid": "#cbd5e1",
    "agent": "#2563eb",
    "target": "#dc2626",
    "distractor": "#f59e0b",
}


def render_frame(agent: tuple[int, int] | None, target: tuple[int, int] = g.TARGET,
                  title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    for r in range(g.SIZE):
        for c in range(g.SIZE):
            color = COLORS["wall"] if g.WALLS[r, c] else COLORS["free"]
            ax.add_patch(plt.Rectangle((c, g.SIZE - 1 - r), 1, 1, facecolor=color,
                                        edgecolor=COLORS["grid"], linewidth=0.6))
    for d in g.DISTRACTORS:
        r, c = d
        ax.add_patch(plt.Circle((c + 0.5, g.SIZE - 1 - r + 0.5), 0.28,
                                 facecolor=COLORS["distractor"], edgecolor="none"))
    tr, tc = target
    ax.add_patch(plt.Circle((tc + 0.5, g.SIZE - 1 - tr + 0.5), 0.32,
                             facecolor=COLORS["target"], edgecolor="none"))
    if agent is not None:
        ar, ac = agent
        ax.add_patch(plt.Circle((ac + 0.5, g.SIZE - 1 - ar + 0.5), 0.3,
                                 facecolor=COLORS["agent"], edgecolor="white", linewidth=1.2))
    ax.set_xlim(0, g.SIZE)
    ax.set_ylim(0, g.SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.3)
    return fig


def frame_png_b64(agent: tuple[int, int] | None, target: tuple[int, int] = g.TARGET,
                   title: str = "") -> str:
    fig = render_frame(agent, target, title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def start_end_images(start: tuple[int, int], target: tuple[int, int] = g.TARGET):
    """Return (start_png_b64, end_png_b64) — the literal 'starting and
    ending images' passed to the LLM judge."""
    return (
        frame_png_b64(start, target, title="start"),
        frame_png_b64(target, target, title="goal"),
    )


def action_field_plot(field_a: dict, field_b: dict, cells: list, title_a: str,
                       title_b: str, out_path: str):
    """Overlay the preferred-action arrow field of two policies (rule/NN vs
    LLM judge) on the map, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=140)
    for ax, field, title in zip(axes, [field_a, field_b], [title_a, title_b]):
        for r in range(g.SIZE):
            for c in range(g.SIZE):
                color = COLORS["wall"] if g.WALLS[r, c] else COLORS["free"]
                ax.add_patch(plt.Rectangle((c, g.SIZE - 1 - r), 1, 1, facecolor=color,
                                            edgecolor=COLORS["grid"], linewidth=0.5))
        tr, tc = g.TARGET
        ax.add_patch(plt.Circle((tc + 0.5, g.SIZE - 1 - tr + 0.5), 0.28,
                                 facecolor=COLORS["target"], edgecolor="none"))
        arrow = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        for cell in cells:
            probs = field.get(str(cell)) or field.get(cell)
            if probs is None:
                continue
            r, c = cell
            best = g.ACTIONS[int(np.argmax(probs))]
            dx, dy = arrow[best]
            conf = max(probs)
            ax.arrow(c + 0.5, g.SIZE - 1 - r + 0.5, dx * 0.32 * conf * 2, dy * 0.32 * conf * 2,
                      head_width=0.14, head_length=0.12, fc="#0f766e", ec="#0f766e")
        ax.set_xlim(0, g.SIZE); ax.set_ylim(0, g.SIZE)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def bar_overlay(labels: list[str], series: dict[str, list[float]], ylabel: str,
                 title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=140)
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def line_plot(x, series: dict[str, list[float]], xlabel: str, ylabel: str, title: str,
              out_path: str, band: dict[str, tuple[list[float], list[float]]] | None = None):
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
    for name, vals in series.items():
        ax.plot(x, vals, marker="o", label=name, linewidth=1.6, markersize=4)
        if band and name in band:
            lo, hi = band[name]
            ax.fill_between(x, lo, hi, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def scatter_calibration(x: list[float], y: list[float], xlabel: str, ylabel: str,
                         title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=140)
    ax.scatter(x, y, alpha=0.55, s=22, color="#2563eb", edgecolor="none")
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#9ca3af", linewidth=1.2,
            label="perfect agreement")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def iteration_histogram(counts: list[int], max_iters: int, xlabel: str, ylabel: str,
                         title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
    bins = np.arange(1, max_iters + 2) - 0.5
    ax.hist(counts, bins=bins, color="#0f766e", edgecolor="white")
    ax.set_xticks(range(1, max_iters + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    frame_png_b64(g.START)
    print("ok")
