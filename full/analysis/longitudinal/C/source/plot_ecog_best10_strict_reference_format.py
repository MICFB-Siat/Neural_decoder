



from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import plot_ecog_all_channels_low_high as ecog


CHANNELS = [31, 0, 63, 15, 25, 86, 19, 84, 40, 113]
DAY_ORDER = [0, 14, 8, 42]
DAY_COLORS = {
    0: "#255aa8",
    14: "#df6b2f",
    8: "#8443a7",
    42: "#2aa179",
}
OUT_DIR = ecog.OUT_DIR.parent / "ecog_speech_best10_reference_format"
Y_LABEL = "normalized response"
ARIAL_FONT_DIR = Path("/home/guoyi/.local/share/fonts/msttcorefonts")
ARIAL_FONT_FILES = (
    ARIAL_FONT_DIR / "Arial.ttf",
    ARIAL_FONT_DIR / "Arial_Bold.ttf",
    ARIAL_FONT_DIR / "Arial_Italic.ttf",
    ARIAL_FONT_DIR / "Arial_Bold_Italic.ttf",
)


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    for font_path in ARIAL_FONT_FILES:
        if not font_path.is_file():
            raise FileNotFoundError(f"Required Arial font is missing: {font_path}")
        font_manager.fontManager.addfont(font_path)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def prepare_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, raw_means, mode_means, n_channels = ecog.load_means()
    keep = t <= (ecog.XLIM_SECONDS + 1e-9)
    t_disp = t[keep]

    low_raw = np.stack([raw_means[(day, ecog.TARGET)] for day in ecog.DAYS], axis=0)
    high_raw = ecog.calibrated_high_all_channels(t, raw_means, mode_means, n_channels)

    low_disp = ecog.row_normalize(ecog.smooth(low_raw[:, :, keep], 7))
    high_disp = ecog.row_normalize(ecog.smooth(high_raw[:, :, keep], 7))
    day_idx = [ecog.DAYS.index(day) for day in DAY_ORDER]
    return t_disp, low_disp[day_idx], high_disp[day_idx]


def style_axis(ax: plt.Axes, ylabel: bool) -> None:
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-0.05, 1.04)
    ax.set_xticks(np.arange(0.0, 2.001, 0.5))
    ax.set_xticklabels([f"{v:.2f}" for v in np.arange(0.0, 2.001, 0.5)])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    if ylabel:
        ax.set_ylabel(Y_LABEL, fontsize=10.5, labelpad=3.0)
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5, pad=1.5)


def plot_panel(ax: plt.Axes, t: np.ndarray, curves: np.ndarray, ylabel: bool, legend: bool) -> None:
    for i, day in enumerate(DAY_ORDER):
        ax.plot(t, curves[i], lw=0.9, color=DAY_COLORS[day], label=f"D{day}")
    style_axis(ax, ylabel=ylabel)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 1, 2, 3]
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="upper right",
            fontsize=7.0,
            handlelength=1.4,
            borderaxespad=0.0,
            labelspacing=0.20,
            ncol=2,
            columnspacing=0.85,
        )


def draw_combined(t: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    n_rows = len(CHANNELS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.35, 20.4), sharex=False, sharey=True)
    fig.subplots_adjust(left=0.165, right=0.985, bottom=0.035, top=0.965, wspace=0.19, hspace=0.50)

    axes[0, 0].set_title("Low-level", fontsize=12.8, fontweight="bold", pad=7.0)
    axes[0, 1].set_title("High-level", fontsize=12.8, fontweight="bold", pad=7.0)

    for row, ch in enumerate(CHANNELS):
        plot_panel(axes[row, 0], t, low[:, ch, :], ylabel=True, legend=False)
        plot_panel(axes[row, 1], t, high[:, ch, :], ylabel=False, legend=True)
        axes[row, 0].text(
            -0.45,
            1.05,
            f"Ch{ch}",
            transform=axes[row, 0].transAxes,
            fontsize=12.0,
            fontweight="bold",
            ha="left",
            va="top",
            color="#202020",
        )

    stem = OUT_DIR / "ecog_speech_best10_reference_format"
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_single_channels(t: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    for ch in CHANNELS:
        fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.12), sharey=True)
        fig.subplots_adjust(left=0.160, right=0.985, bottom=0.155, top=0.82, wspace=0.19)
        axes[0].set_title("Low-level", fontsize=12.8, fontweight="bold", pad=7.0)
        axes[1].set_title("High-level", fontsize=12.8, fontweight="bold", pad=7.0)
        plot_panel(axes[0], t, low[:, ch, :], ylabel=True, legend=False)
        plot_panel(axes[1], t, high[:, ch, :], ylabel=False, legend=True)
        axes[0].text(
            -0.32,
            1.30,
            f"Ch{ch}",
            transform=axes[0].transAxes,
            fontsize=15.0,
            fontweight="bold",
            ha="left",
            va="top",
            color="#202020",
        )
        stem = OUT_DIR / f"ecog_speech_ch{ch:03d}_reference_format"
        fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def write_readme() -> None:
    lines = [
        "# ECoG Speech Best 10 Reference Format",
        "",
        "Channels: " + ", ".join(f"Ch{ch}" for ch in CHANNELS),
        "",
        "Trial category: Back trials (label index 0).",
        "",
        "Layout matches the provided reference: Low-level and High-level columns, normalized response y-axis, 0-2 s x-axis, and D0/D14/D8/D42 legend colors.",
        "",
        "Only SVG and PDF files are saved.",
    ]
    pass
    rows = ["channel,trial_category,label_index\n"]
    rows.extend(f"Ch{ch},Back,0\n" for ch in CHANNELS)
    (OUT_DIR / "selected_channels_trial_category.csv").write_text("".join(rows), encoding="utf-8")


def main() -> None:
    setup()
    t, low, high = prepare_curves()
    draw_combined(t, low, high)
    draw_single_channels(t, low, high)
    write_readme()
    print(f"saved to {OUT_DIR}")
    print("channels:", CHANNELS)


if __name__ == "__main__":
    main()
