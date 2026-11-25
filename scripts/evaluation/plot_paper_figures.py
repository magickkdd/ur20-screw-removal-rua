"""Generate publication figures from the tracked paired-replay result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "outputs" / "paper_figures"

COLORS = {
    "navy": "#264653",
    "blue": "#457B9D",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "yellow": "#E9C46A",
    "gray": "#6B7280",
    "light": "#F3F6F8",
    "ink": "#1F2933",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax: plt.Axes, x: float, y: float, width: float, height: float,
            label: str, face: str, edge: str | None = None,
            text_color: str = COLORS["ink"], fontsize: float = 10) -> None:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.5, facecolor=face, edgecolor=edge or face,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            color=text_color, fontsize=fontsize, linespacing=1.35,
            transform=ax.transAxes)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = COLORS["gray"], connection: str = "arc3,rad=0") -> None:
    ax.annotate(
        "", xy=end, xytext=start, xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": color,
                    "connectionstyle": connection, "shrinkA": 2, "shrinkB": 2},
    )


def figure_method(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_box(ax, 0.02, 0.45, 0.16, 0.24, "RGB images +\npublic robot state\n\nlatest observation",
            COLORS["light"], COLORS["blue"])
    add_box(ax, 0.23, 0.66, 0.22, 0.17, "Native policy\nmove_to: one waypoint / turn",
            "#EAF1F6", COLORS["blue"])
    add_box(ax, 0.23, 0.33, 0.22, 0.19, "Fused policy\nmove_sequence: up to 4\nfree-space waypoints / turn",
            "#E8F4F1", COLORS["teal"])
    add_box(ax, 0.51, 0.45, 0.20, 0.24, "Robot adapter\n\nexecute each waypoint\nwith existing motion limits",
            "#F8F1E5", COLORS["yellow"])
    add_box(ax, 0.76, 0.49, 0.22, 0.22, "Stage guard after each point\n\ncontact | vision < 0.7\nXY alignment phase change",
            "#FCEDE9", COLORS["orange"])
    add_box(ax, 0.54, 0.10, 0.42, 0.20,
            "Unbatched task controllers\nvisual alignment → compliant approach\nspiral search + tactile seating → extraction",
            "#EAF0F2", COLORS["navy"])

    arrow(ax, (0.18, 0.59), (0.23, 0.74), COLORS["blue"])
    arrow(ax, (0.18, 0.53), (0.23, 0.43), COLORS["teal"])
    arrow(ax, (0.45, 0.74), (0.51, 0.61), COLORS["blue"])
    arrow(ax, (0.45, 0.43), (0.51, 0.52), COLORS["teal"])
    arrow(ax, (0.71, 0.57), (0.76, 0.59), COLORS["orange"])
    arrow(ax, (0.86, 0.49), (0.79, 0.30), COLORS["navy"])
    arrow(ax, (0.54, 0.18), (0.11, 0.45), COLORS["gray"], "arc3,rad=-0.22")

    ax.text(0.02, 0.92, "Guarded action-stream control for UR20 screw extraction",
            fontsize=15, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.02, 0.025,
            "Force control, tactile state updates, and force guards run every physics step; diagnostic traces are sampled every 25 steps.",
            fontsize=9, color=COLORS["gray"], transform=ax.transAxes)
    save_figure(fig, output, "fig1_method_overview")


def paired_bars(ax: plt.Axes, rows: dict[str, dict[str, float]], ylabel: str,
                title: str, value_format: str = ".1f", ylim: tuple[float, float] | None = None) -> None:
    labels = list(rows)
    x = list(range(len(labels)))
    width = 0.34
    native = [rows[key]["native"] for key in labels]
    fused = [rows[key]["fused"] for key in labels]
    bars_a = ax.bar([i - width / 2 for i in x], native, width,
                    label="Native", color=COLORS["blue"])
    bars_b = ax.bar([i + width / 2 for i in x], fused, width,
                    label="Fused", color=COLORS["orange"])
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", color="#D9E0E5", linewidth=0.7)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    span = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018
    for bars in (bars_a, bars_b):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + span,
                    format(value, value_format), ha="center", va="bottom", fontsize=8)


def figure_action_stream(result: dict, output: Path) -> None:
    records = result["results"]
    sources = ["agentview", "wristview"]

    def metric(name: str) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        for source in sources:
            values[source] = {
                row["mode"]: float(row[name])
                for row in records if source in row["source_trace"]
            }
        return values

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.1))
    paired_bars(axes[0, 0], metric("policy_turns"), "Policy turns", "(a) Policy calls",
                ".0f", (0, 36))
    paired_bars(axes[0, 1], metric("replay_wall_s"), "Replay wall time (s)",
                "(b) Fast-forward simulation", ".1f", (0, 175))
    paired_bars(axes[1, 0], metric("archived_policy_latency_s"),
                "Archived response latency (s)", "(c) Policy waiting reconstructed from logs",
                ".0f", (0, 1400))
    paired_bars(axes[1, 1], metric("projected_total_s"),
                "Latency-reconstructed duration (s)", "(d) Reconstructed total",
                ".0f", (0, 1500))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0, w_pad=2.0)
    save_figure(fig, output, "fig2_action_stream_results")


def figure_trace_optimization(result: dict, output: Path) -> None:
    before = {row["mode"]: row for row in result["baseline"]["modes"]}
    after = {row["mode"]: row for row in result["sparse_diagnostics"]["modes"]}
    modes = ["native", "fused"]
    x = [0, 1]
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    panels = [
        ("replay_wall_s", "Replay wall time (s)", "(a) Simulation runtime"),
        ("trace_bytes", "Force-trace size (MB)", "(b) Diagnostic output volume"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        if key == "trace_bytes":
            old_values = [before[mode][key] / 1_000_000 for mode in modes]
            new_values = [after[mode][key] / 1_000_000 for mode in modes]
            value_fmt = ".1f"
        else:
            old_values = [before[mode][key] for mode in modes]
            new_values = [after[mode][key] for mode in modes]
            value_fmt = ".1f"
        old_bars = ax.bar([i - width / 2 for i in x], old_values, width,
                          color="#9AA7B2", label="Every-step diagnostics")
        new_bars = ax.bar([i + width / 2 for i in x], new_values, width,
                          color=COLORS["teal"], label="25-step sampling")
        ax.set_xticks(x, ["Native", "Fused"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="y", color="#D9E0E5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(old_values) * 1.23)
        span = (max(old_values) - min(new_values)) * 0.035
        for bars in (old_bars, new_bars):
            for bar in bars:
                value = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, value + span,
                        format(value, value_fmt), ha="center", va="bottom", fontsize=9)
        for index, mode in enumerate(modes):
            reduction = 100 * (1 - after[mode][key] / before[mode][key])
            label = f"−{reduction:.1f}%"
            ymax = max(old_values[index], new_values[index])
            ax.text(index, ymax + span * 2.2, label, ha="center", va="bottom",
                    color=COLORS["navy"], weight="bold", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.2)
    save_figure(fig, output, "fig3_sparse_diagnostic_results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    action_result = load_json(RESULTS / "action_stream_fast_replay.json")
    trace_result = load_json(RESULTS / "action_stream_trace_optimization.json")
    figure_method(args.output_dir)
    figure_action_stream(action_result, args.output_dir)
    figure_trace_optimization(trace_result, args.output_dir)


if __name__ == "__main__":
    main()
