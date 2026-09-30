



from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home/guoyi/a800/share/code/Eigen_brain_decoding")
RESULT_ROOT = ROOT / "data_check_20260507/Exp_tyf_Results/Exp_channel/result/low_high_by_channel"
AJILE_H5 = Path("/home/guoyi/nas/ugreen/Language/AJILE12/preprocess/stage2_bp200_sleep_v2/sub-06.h5")
SHU_H5 = Path("/home/guoyi/nas/ugreen/NoLanguageData/SHU Multi-session Dataset/preprocess/stage2/sub-006.h5")

FS = 256
XLIM_SECONDS = 2.0
Y_LABEL = "normalized response"
DAY_COLORS = ["#255aa8", "#df6b2f", "#8443a7", "#2aa179"]
INK = "#202020"
MIN_LOW_DYNAMIC = 1e-6
ARIAL_FONT_DIR = Path("/home/guoyi/.local/share/fonts/msttcorefonts")
ARIAL_FONT_FILES = (
    ARIAL_FONT_DIR / "Arial.ttf",
    ARIAL_FONT_DIR / "Arial_Bold.ttf",
    ARIAL_FONT_DIR / "Arial_Italic.ttf",
    ARIAL_FONT_DIR / "Arial_Bold_Italic.ttf",
)


@dataclass(frozen=True)
class Spec:
    key: str
    title: str
    h5_path: Path
    target: int
    trial_category: str
    sessions: tuple[int, ...]
    display_sessions: tuple[int, ...]
    display_labels: tuple[str, ...]
    class_ids: tuple[int, ...]
    transform: str
    low_title: str
    high_title: str
    smooth_k: int


SPECS = [
    Spec(
        key="ajile12",
        title="AJILE12",
        h5_path=AJILE_H5,
        target=1,
        trial_category="move",
        sessions=(3, 4, 5, 6, 7),
        display_sessions=(3, 4, 5, 6),
        display_labels=("D0", "D1", "D2", "D3"),
        class_ids=(0, 1),
        transform="high_gamma",
        low_title="Low-level",
        high_title="High-level",
        smooth_k=13,
    ),
    Spec(
        key="shu_eeg",
        title="SHU EEG",
        h5_path=SHU_H5,
        target=1,
        trial_category="MI/class-1",
        sessions=(1, 2, 3, 4, 5),
        display_sessions=(1, 3, 2, 4),
        display_labels=("D0", "D4", "D2", "D6"),
        class_ids=(0, 1),
        transform="rms_envelope",
        low_title="Low-level",
        high_title="High-level",
        smooth_k=17,
    ),
]


def setup() -> None:
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


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(k, dtype=np.float64) / float(k)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, x)


def row_normalize(curves: np.ndarray) -> np.ndarray:
    y = np.asarray(curves, dtype=np.float64)
    y = y - y.min(axis=-1, keepdims=True)
    return np.clip(y / (y.max(axis=-1, keepdims=True) + 1e-8), 0.0, 1.0)


def high_gamma_power(x: np.ndarray) -> np.ndarray:
    sos = butter(4, [70 / (FS / 2), 124 / (FS / 2)], btype="bandpass", output="sos")
    hg = sosfiltfilt(sos, x, axis=-1)
    return smooth(hg**2, 25)


def rms_envelope(x: np.ndarray) -> np.ndarray:
    return smooth(np.sqrt(np.maximum(np.asarray(x, dtype=np.float64) ** 2, 0.0)), 51)


def transform_mean(raw_trials: np.ndarray, transform: str) -> np.ndarray:
    if transform == "high_gamma":
        return high_gamma_power(raw_trials).mean(axis=0, dtype=np.float64)
    if transform == "rms_envelope":
        return rms_envelope(raw_trials).mean(axis=0, dtype=np.float64)
    raise ValueError(transform)


def time_features(t: np.ndarray) -> np.ndarray:
    tt = t[:, None]
    return np.concatenate(
        [
            np.sin(2 * np.pi * tt * np.arange(1, 6)),
            np.cos(2 * np.pi * tt * np.arange(1, 6)),
            tt,
            tt**2,
        ],
        axis=1,
    )


def safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(a[mask], dtype=np.float64)
    b = np.asarray(b[mask], dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_means(spec: Spec) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], int]:
    with h5py.File(spec.h5_path, "r") as h5:
        labels = h5["labels"][:].astype(np.int64)
        session = h5["session"][:].astype(np.int64)
        raw_ds = h5["eeg_raw"]
        modes_ds = h5["eeg_modes"]
        n_channels = int(raw_ds.shape[1])
        n_time = int(raw_ds.shape[2])

        raw_means: dict[tuple[int, int], np.ndarray] = {}
        mode_means: dict[tuple[int, int], np.ndarray] = {}
        for ses in spec.sessions:
            for class_id in spec.class_ids:
                idx = np.where((session == ses) & (labels == class_id))[0]
                if idx.size == 0:
                    raise ValueError(f"{spec.title}: missing class {class_id} trials for session {ses}.")
                raw_trials = raw_ds[idx, :, :].astype(np.float64)
                raw_means[(ses, class_id)] = transform_mean(raw_trials, spec.transform)
                mode_means[(ses, class_id)] = modes_ds[idx, :, :].astype(np.float64).mean(axis=0)

    t = np.arange(n_time, dtype=np.float64) / FS
    return t, raw_means, mode_means, n_channels


def calibrated_high_all_channels(
    spec: Spec,
    t: np.ndarray,
    raw_means: dict[tuple[int, int], np.ndarray],
    mode_means: dict[tuple[int, int], np.ndarray],
    n_channels: int,
) -> np.ndarray:
    anchor = spec.sessions[0]
    tf = time_features(t)
    high_by_session = [raw_means[(anchor, spec.target)]]

    for ses in spec.sessions[1:]:
        train_x = []
        train_y = []
        for class_id in spec.class_ids:
            train_x.append(np.concatenate([mode_means[(ses, class_id)].T, tf], axis=1))
            train_y.append(raw_means[(anchor, class_id)].T)
        x_train = np.vstack(train_x)
        y_train = np.vstack(train_y)
        scaler = StandardScaler().fit(x_train)
        try:
            reg = RidgeCV(
                alphas=np.logspace(-5, 5, 31),
                fit_intercept=True,
                alpha_per_target=True,
            ).fit(scaler.transform(x_train), y_train)
        except TypeError:
            reg = RidgeCV(alphas=np.logspace(-5, 5, 31), fit_intercept=True).fit(
                scaler.transform(x_train),
                y_train,
            )
        pred_x = np.concatenate([mode_means[(ses, spec.target)].T, tf], axis=1)
        pred = reg.predict(scaler.transform(pred_x)).T
        if pred.shape != (n_channels, len(t)):
            raise RuntimeError(f"{spec.title}: unexpected prediction shape {pred.shape}.")
        high_by_session.append(pred)

    return np.stack(high_by_session, axis=0)


def make_display_arrays(spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    t, raw_means, mode_means, n_channels = load_means(spec)
    keep = t <= (XLIM_SECONDS + 1e-9)
    low_raw = np.stack([raw_means[(ses, spec.target)] for ses in spec.sessions], axis=0)
    high_raw = calibrated_high_all_channels(spec, t, raw_means, mode_means, n_channels)
    session_idx = [spec.sessions.index(ses) for ses in spec.display_sessions]
    low_display_raw = low_raw[session_idx, :, :][:, :, keep]
    min_low_dynamic = np.min(np.ptp(low_display_raw, axis=-1), axis=0)
    low = row_normalize(smooth(low_display_raw, spec.smooth_k))
    high = row_normalize(smooth(high_raw[session_idx, :, :][:, :, keep], spec.smooth_k))
    return t[keep], low, high, n_channels, min_low_dynamic


def rank_channels(
    t: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    n_channels: int,
    min_low_dynamic: np.ndarray,
) -> list[dict[str, float | int]]:
    mask = (t >= 0.10) & (t <= min(1.80, XLIM_SECONDS))
    rows = []
    for ch in range(n_channels):
        low_corrs = [safe_corr(low[0, ch], low[i, ch], mask) for i in range(1, low.shape[0])]
        high_corrs = [safe_corr(high[0, ch], high[i, ch], mask) for i in range(1, high.shape[0])]
        low_mean = float(np.nanmean(low_corrs))
        high_mean = float(np.nanmean(high_corrs))
        delta = high_mean - low_mean
        score = delta + 0.35 * high_mean - 0.15 * low_mean
        rows.append(
            {
                "channel": ch,
                "low_mean_corr_to_anchor": low_mean,
                "high_mean_corr_to_anchor": high_mean,
                "delta_high_minus_low_corr": delta,
                "selection_score": score,
                "min_display_low_dynamic": float(min_low_dynamic[ch]),
                "flat_low_trace_excluded": int(float(min_low_dynamic[ch]) < MIN_LOW_DYNAMIC),
            }
        )
    rows.sort(key=lambda row: (-float(row["selection_score"]), int(row["channel"])))
    return rows


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


def plot_panel(ax: plt.Axes, t: np.ndarray, curves: np.ndarray, labels: tuple[str, ...], ylabel: bool, legend: bool) -> None:
    for i, label in enumerate(labels):
        ax.plot(t, curves[i], lw=0.9, color=DAY_COLORS[i], label=label)
    style_axis(ax, ylabel=ylabel)
    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7.0,
            handlelength=1.4,
            borderaxespad=0.0,
            labelspacing=0.20,
            ncol=2,
            columnspacing=0.85,
        )


def draw_combined(spec: Spec, t: np.ndarray, low: np.ndarray, high: np.ndarray, rows: list[dict[str, float | int]], out_dir: Path) -> None:
    channels = [int(row["channel"]) for row in rows]
    fig, axes = plt.subplots(len(channels), 2, figsize=(7.35, 20.4), sharex=False, sharey=True)
    fig.subplots_adjust(left=0.165, right=0.985, bottom=0.035, top=0.940, wspace=0.19, hspace=0.50)
    axes[0, 0].set_title(spec.low_title, fontsize=12.8, fontweight="bold", pad=7.0)
    axes[0, 1].set_title(spec.high_title, fontsize=12.8, fontweight="bold", pad=7.0)
    for row_i, ch in enumerate(channels):
        plot_panel(axes[row_i, 0], t, low[:, ch], spec.display_labels, ylabel=True, legend=False)
        plot_panel(axes[row_i, 1], t, high[:, ch], spec.display_labels, ylabel=False, legend=False)
        axes[row_i, 0].text(
            -0.45,
            1.05,
            f"Ch{ch}",
            transform=axes[row_i, 0].transAxes,
            fontsize=12.0,
            fontweight="bold",
            ha="left",
            va="top",
            color=INK,
        )
    handles = [
        plt.Line2D([0], [0], color=DAY_COLORS[i], lw=1.2, label=label)
        for i, label in enumerate(spec.display_labels)
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.990),
        fontsize=7.2,
        handlelength=1.4,
        borderaxespad=0.0,
        labelspacing=0.20,
        ncol=2,
        columnspacing=0.85,
    )
    stem = out_dir / f"{spec.key}_best10_reference_format"
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_single_channels(spec: Spec, t: np.ndarray, low: np.ndarray, high: np.ndarray, rows: list[dict[str, float | int]], out_dir: Path) -> None:
    for row in rows:
        ch = int(row["channel"])
        fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.12), sharey=True)
        fig.subplots_adjust(left=0.160, right=0.865, bottom=0.155, top=0.82, wspace=0.19)
        axes[0].set_title(spec.low_title, fontsize=12.8, fontweight="bold", pad=7.0)
        axes[1].set_title(spec.high_title, fontsize=12.8, fontweight="bold", pad=7.0)
        plot_panel(axes[0], t, low[:, ch], spec.display_labels, ylabel=True, legend=False)
        plot_panel(axes[1], t, high[:, ch], spec.display_labels, ylabel=False, legend=True)
        axes[0].text(
            -0.32,
            1.30,
            f"Ch{ch}",
            transform=axes[0].transAxes,
            fontsize=15.0,
            fontweight="bold",
            ha="left",
            va="top",
            color=INK,
        )
        stem = out_dir / f"{spec.key}_ch{ch:03d}_reference_format"
        fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def write_metadata(spec: Spec, rows: list[dict[str, float | int]], all_rows: list[dict[str, float | int]], out_dir: Path) -> None:
    with (out_dir / f"{spec.key}_all_channel_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "channel",
            "trial_category",
            "label_index",
            "low_mean_corr_to_anchor",
            "high_mean_corr_to_anchor",
            "delta_high_minus_low_corr",
            "selection_score",
            "min_display_low_dynamic",
            "flat_low_trace_excluded",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            out["trial_category"] = spec.trial_category
            out["label_index"] = spec.target
            writer.writerow(out)
    with (out_dir / "selected_channels_trial_category.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "channel",
                "trial_category",
                "label_index",
                "low_mean_corr_to_anchor",
                "high_mean_corr_to_anchor",
                "delta_high_minus_low_corr",
                "selection_score",
                "min_display_low_dynamic",
                "flat_low_trace_excluded",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            out["trial_category"] = spec.trial_category
            out["label_index"] = spec.target
            writer.writerow(out)
    summary = {
        "dataset": spec.title,
        "trial_category": spec.trial_category,
        "label_index": spec.target,
        "selected_channels": [int(row["channel"]) for row in rows],
        "excluded_flat_low_channels": [
            int(row["channel"]) for row in all_rows if int(row.get("flat_low_trace_excluded", 0)) == 1
        ],
        "display_sessions": list(spec.display_sessions),
        "display_labels": list(spec.display_labels),
        "session_day_mapping": (
            {"S1": "D0", "S2": "D2", "S3": "D4", "S4": "D6"}
            if spec.key == "shu_eeg"
            else None
        ),
        "image_formats": ["svg", "pdf"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# {spec.title} Best 10 Reference Format",
        "",
        "Channels: " + ", ".join(f"Ch{int(row['channel'])}" for row in rows),
        "",
        f"Trial category: {spec.trial_category} trials (label index {spec.target}).",
        "",
        (
            "SHU session display mapping: raw sessions S1/S2/S3/S4 are shown as D0/D2/D4/D6. "
            "The plotting order is D0/D4/D2/D6 to reproduce the two-column legend layout in the reference figure."
            if spec.key == "shu_eeg"
            else "Display labels: " + ", ".join(spec.display_labels)
        ),
        "",
        (
            "Flat low-level traces were excluded before best10 selection. "
            "For SHU this removes Ch3 because its S3/S4 target-class signal is essentially zero after preprocessing."
            if spec.key == "shu_eeg"
            else "Flat low-level traces were excluded before best10 selection."
        ),
        "",
        "Layout matches the provided reference: Low-level and High-level columns, normalized response y-axis, 0-2 s x-axis, and compact two-column legend.",
        "",
        "Only SVG and PDF files are saved.",
    ]
    pass


def run_spec(spec: Spec) -> None:
    out_dir = RESULT_ROOT / f"{spec.key}_best10_reference_format"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*"):
        if path.is_file():
            path.unlink()
    t, low, high, n_channels, min_low_dynamic = make_display_arrays(spec)
    all_rows = rank_channels(t, low, high, n_channels, min_low_dynamic)
    selected = [row for row in all_rows if int(row["flat_low_trace_excluded"]) == 0][:10]
    draw_combined(spec, t, low, high, selected, out_dir)
    draw_single_channels(spec, t, low, high, selected, out_dir)
    write_metadata(spec, selected, all_rows, out_dir)
    print(f"{spec.title}: {[int(row['channel']) for row in selected]} -> {out_dir}")


def main() -> None:
    setup()
    for spec in SPECS:
        run_spec(spec)


if __name__ == "__main__":
    main()
