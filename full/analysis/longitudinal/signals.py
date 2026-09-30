











from __future__ import annotations

import argparse
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
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


from resources import path as resource_path
WORK_DIR = Path(__file__).resolve().parent
OUT_DIR = WORK_DIR / 'outputs/signals'
AJILE_H5 = resource_path('ajile_h5')
SHU_H5 = resource_path('shu_h5')
ECOG_H5 = resource_path('ecog_h5')

FS = 256
XLIM_SECONDS = 2.0
DAY_COLORS = ["#1b4f9c", "#d95f02", "#009e73", "#7b3294", "#4c78a8", "#9d755d"]
INK = "#20242a"
MUTED = "#6f767d"
GRID = "#d7dadd"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    h5_path: Path
    target: int
    target_name: str
    days: tuple[int, ...]
    day_prefix: str
    class_ids: tuple[int, ...]
    transform: str
    current_channels: tuple[int, ...]
    n_select: int
    holdout_target_for_high: bool
    smooth_k: int
    win: tuple[float, float]
    low_label: str
    high_label: str


SPECS = [
    DatasetSpec(
        key="ajile12",
        title="AJILE12",
        h5_path=AJILE_H5,
        target=1,
        target_name="move",
        days=(3, 4, 5, 6, 7),
        day_prefix="D",
        class_ids=(0, 1),
        transform="high_gamma",
        current_channels=(52,),
        n_select=8,
        holdout_target_for_high=False,
        smooth_k=13,
        win=(0.65, 1.35),
        low_label="Low-level high-gamma",
        high_label="Mode-calibrated high-level output",
    ),
    DatasetSpec(
        key="shu_eeg",
        title="SHU EEG",
        h5_path=SHU_H5,
        target=1,
        target_name="MI/class-1",
        days=(1, 2, 3, 4, 5),
        day_prefix="S",
        class_ids=(0, 1),
        transform="rms_envelope",
        current_channels=(17,),
        n_select=8,
        holdout_target_for_high=False,
        smooth_k=17,
        win=(0.70, 1.50),
        low_label="Low-level temporal envelope",
        high_label="Mode-calibrated high-level output",
    ),
    DatasetSpec(
        key="ecog_speech",
        title="ECoG Speech",
        h5_path=ECOG_H5,
        target=0,
        target_name="Back",
        days=(0, 8, 14, 42),
        day_prefix="D",
        class_ids=(0, 1, 2, 3, 4, 5),
        transform="raw",
        current_channels=(31, 125, 123),
        n_select=8,
        holdout_target_for_high=True,
        smooth_k=7,
        win=(0.65, 1.35),
        low_label="Low-level raw high-gamma",
        high_label="Leave-class-out calibrated high-level output",
    ),
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(k, dtype=np.float64) / float(k)
    x = np.asarray(x, dtype=np.float64)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, x)


def high_gamma_power(x: np.ndarray, fs: int = FS) -> np.ndarray:
    sos = butter(4, [70 / (fs / 2), 124 / (fs / 2)], btype="bandpass", output="sos")
    hg = sosfiltfilt(sos, x, axis=-1)
    return smooth(hg**2, 25)


def rms_envelope(x: np.ndarray, k: int = 41) -> np.ndarray:
    return smooth(np.sqrt(np.maximum(np.asarray(x, dtype=np.float64) ** 2, 0.0)), k)


def transform_mean(raw_trials: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return raw_trials.mean(axis=0, dtype=np.float64)
    if transform == "high_gamma":
        return high_gamma_power(raw_trials, FS).mean(axis=0, dtype=np.float64)
    if transform == "rms_envelope":
        return rms_envelope(raw_trials, k=51).mean(axis=0, dtype=np.float64)
    raise ValueError(f"Unknown transform: {transform}")


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


def template_normalize(
    curves: np.ndarray,
    t: np.ndarray,
    ref_curve: np.ndarray,
    win: tuple[float, float],
    smooth_k: int,
) -> np.ndarray:
    curves = smooth(np.asarray(curves, dtype=np.float64), smooth_k)
    ref_curve = smooth(np.asarray(ref_curve, dtype=np.float64), smooth_k)
    base = t <= (t[-1] * 0.20)
    centered = curves - curves[:, base].mean(axis=1, keepdims=True)
    ref_centered = ref_curve - ref_curve[base].mean()
    mask = (t >= win[0]) & (t <= win[1])
    denom = np.percentile(ref_centered[mask], 95)
    if abs(float(denom)) < 1e-8:
        denom = np.percentile(np.abs(ref_centered), 95) + 1e-8
    return np.clip(centered / float(denom), -0.30, 1.45)


def row_normalize(curves: np.ndarray) -> np.ndarray:
    y = np.asarray(curves, dtype=np.float64)
    y = y - y.min(axis=1, keepdims=True)
    den = y.max(axis=1, keepdims=True)
    return np.clip(y / (den + 1e-8), 0.0, 1.0)


def safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_day_class_means(spec: DatasetSpec) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], int]:
    if not spec.h5_path.is_file():
        raise FileNotFoundError(spec.h5_path)
    with h5py.File(spec.h5_path, "r") as h5:
        labels = h5["labels"][:].astype(np.int64)
        session = h5["session"][:].astype(np.int64)
        raw_ds = h5["eeg_raw"]
        modes_ds = h5["eeg_modes"]
        n_channels = int(raw_ds.shape[1])
        n_time = int(raw_ds.shape[2])

        raw_means: dict[tuple[int, int], np.ndarray] = {}
        mode_means: dict[tuple[int, int], np.ndarray] = {}
        for day in spec.days:
            for class_id in spec.class_ids:
                idx = np.where((session == day) & (labels == class_id))[0]
                if idx.size == 0:
                    continue
                raw_trials = raw_ds[idx, :, :].astype(np.float64)
                raw_means[(day, class_id)] = transform_mean(raw_trials, spec.transform)
                mode_means[(day, class_id)] = modes_ds[idx, :, :].astype(np.float64).mean(axis=0)

    t = np.arange(n_time, dtype=np.float64) / FS
    return t, raw_means, mode_means, n_channels


def normalize_for_display(
    curves: np.ndarray,
    t: np.ndarray,
    ref_curve: np.ndarray,
    spec: DatasetSpec,
) -> np.ndarray:
    keep = t <= (XLIM_SECONDS + 1e-9)
    del ref_curve
    smoothed = smooth(np.asarray(curves, dtype=np.float64), spec.smooth_k)
    return row_normalize(smoothed[:, keep])


def rank_channels(
    spec: DatasetSpec,
    t: np.ndarray,
    raw_means: dict[tuple[int, int], np.ndarray],
    n_channels: int,
) -> list[dict[str, float | int]]:
    keep = t <= (XLIM_SECONDS + 1e-9)
    corr_mask = keep & (t >= 0.10) & (t <= min(1.80, XLIM_SECONDS))
    rows = []
    for ch in range(n_channels):
        low_raw = np.asarray([raw_means[(day, spec.target)][ch] for day in spec.days])
        ref_raw = low_raw[0]
        low_disp = normalize_for_display(low_raw, t, ref_raw, spec)
        ref = low_disp[0]
        late_corrs = [safe_corr(ref, low_disp[i], corr_mask[keep]) for i in range(1, len(spec.days))]
        finite = [v for v in late_corrs if np.isfinite(v)]
        mean_corr = float(np.mean(finite)) if finite else float("nan")
        dynamic = float(np.percentile(ref_raw[keep], 95) - np.percentile(ref_raw[keep], 5))
        drift = 1.0 - mean_corr if np.isfinite(mean_corr) else 0.0
        score = dynamic * max(drift, 0.0)
        rows.append(
            {
                "channel": ch,
                "low_mean_corr_to_anchor": mean_corr,
                "anchor_dynamic_range": dynamic,
                "low_drift_score": float(score),
            }
        )
    rows.sort(key=lambda row: (-float(row["low_drift_score"]), int(row["channel"])))
    return rows


def selected_channels(spec: DatasetSpec, ranking: list[dict[str, float | int]], n_channels: int) -> list[int]:
    ranked = [int(row["channel"]) for row in ranking]
    requested = [ch for ch in spec.current_channels if 0 <= ch < n_channels]
    return unique_in_order([*requested, *ranked])[: spec.n_select]


def make_channel_curves(
    spec: DatasetSpec,
    channel: int,
    t: np.ndarray,
    raw_means: dict[tuple[int, int], np.ndarray],
    mode_means: dict[tuple[int, int], np.ndarray],
) -> dict:
    anchor_day = spec.days[0]
    ref_raw = raw_means[(anchor_day, spec.target)][channel]
    tf = time_features(t)
    low_raw = []
    high_raw = []

    for day in spec.days:
        low_raw.append(raw_means[(day, spec.target)][channel])
        if day == anchor_day:
            high_raw.append(ref_raw)
            continue

        train_x = []
        train_y = []
        for class_id in spec.class_ids:
            if spec.holdout_target_for_high and class_id == spec.target:
                continue
            if (day, class_id) not in mode_means or (anchor_day, class_id) not in raw_means:
                continue
            x_c = mode_means[(day, class_id)].T
            y_c = raw_means[(anchor_day, class_id)][channel]
            train_x.append(np.concatenate([x_c, tf], axis=1))
            train_y.append(y_c)

        if not train_x:
            raise RuntimeError(f"No high-level training data for {spec.title} channel {channel}, day {day}.")

        x_train = np.vstack(train_x)
        y_train = np.concatenate(train_y)
        scaler = StandardScaler().fit(x_train)
        reg = RidgeCV(alphas=np.logspace(-5, 5, 31), fit_intercept=True).fit(
            scaler.transform(x_train),
            y_train,
        )
        pred_x = np.concatenate([mode_means[(day, spec.target)].T, tf], axis=1)
        high_raw.append(reg.predict(scaler.transform(pred_x)))

    low_raw_arr = np.asarray(low_raw)
    high_raw_arr = np.asarray(high_raw)
    keep = t <= (XLIM_SECONDS + 1e-9)
    low_disp = normalize_for_display(low_raw_arr, t, ref_raw, spec)
    high_disp = normalize_for_display(high_raw_arr, t, ref_raw, spec)
    t_disp = t[keep]
    corr_mask = (t_disp >= 0.10) & (t_disp <= min(1.80, XLIM_SECONDS))
    low_corrs = [safe_corr(low_disp[0], low_disp[i], corr_mask) for i in range(1, len(spec.days))]
    high_corrs = [safe_corr(high_disp[0], high_disp[i], corr_mask) for i in range(1, len(spec.days))]
    low_mean = float(np.nanmean(low_corrs))
    high_mean = float(np.nanmean(high_corrs))
    return {
        "channel": channel,
        "t": t_disp,
        "low": low_disp,
        "high": high_disp,
        "low_mean_corr": low_mean,
        "high_mean_corr": high_mean,
        "delta_corr": high_mean - low_mean,
    }


def style_axis(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.axhline(0.0, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(0.0, XLIM_SECONDS)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks(np.arange(0.0, XLIM_SECONDS + 0.001, 0.5))
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(length=3.0, width=0.7, pad=2.0)
    ax.set_xlabel("time within trial (s)", labelpad=2.0)
    ax.set_ylabel("trace-normalized response" if show_ylabel else "", labelpad=2.0)


def draw_channel_figure(spec: DatasetSpec, curves: dict, out_dir: Path) -> None:
    labels = [f"{spec.day_prefix}{day}" for day in spec.days]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    fig.subplots_adjust(left=0.082, right=0.986, bottom=0.19, top=0.75, wspace=0.18)

    for ax_idx, (ax, key, panel_title) in enumerate(
        [
            (axes[0], "low", f"{spec.low_label}: Ch{curves['channel']}"),
            (axes[1], "high", spec.high_label),
        ]
    ):
        data = curves[key]
        for i, label in enumerate(labels):
            ax.plot(
                curves["t"],
                data[i],
                color=DAY_COLORS[i % len(DAY_COLORS)],
                lw=1.95 if i == 0 else 1.35,
                alpha=0.98 if i == 0 else 0.92,
                label=label,
            )
        style_axis(ax, show_ylabel=(ax_idx == 0))
        ax.set_title(panel_title, fontsize=7.8, fontweight="bold", pad=3.0, color=INK)

    axes[1].legend(loc="upper right", fontsize=6.1, handlelength=1.4, labelspacing=0.25, borderaxespad=0.0)
    fig.text(
        0.082,
        0.965,
        f"{spec.title} Ch{curves['channel']}: low-level vs high-level temporal traces",
        fontsize=8.9,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    fig.text(
        0.082,
        0.885,
        (
            f"class-averaged {spec.target_name}; "
            f"mean corr to anchor: low {curves['low_mean_corr']:.2f}, high {curves['high_mean_corr']:.2f}"
        ),
        fontsize=6.4,
        ha="left",
        va="top",
        color=MUTED,
    )

    stem = out_dir / f"{spec.key}_ch{int(curves['channel']):03d}_low_high"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    selected: dict[str, list[int]],
    ranking_summary: dict[str, list[dict[str, float | int]]],
    metric_rows: list[dict[str, object]],
) -> None:
    with (out_dir / "selected_channels.json").open("w", encoding="utf-8") as f:
        json.dump({"selected": selected, "ranking_top20": ranking_summary}, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "dataset",
        "channel",
        "selected_channel_order",
        "low_drift_score",
        "low_rank_mean_corr",
        "high_reconstruction_mean_corr",
        "delta_high_minus_low_corr",
        "anchor_dynamic_range",
        "current_channel_from_previous_panel",
        "figure_png",
    ]
    with (out_dir / "channel_low_high_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = [
        "# Per-Channel Low-vs-High Temporal Traces",
        "",
        "One channel is saved as one figure. Each figure has the low-level trace panel on the left and the calibrated high-level trace panel on the right.",
        "",
        "The right panel follows the previous drift_figs logic: it is not a raw late-day high-level dimension. It is reconstructed/calibrated from eigenmode features into the anchor-day channel-response space.",
        "",
        "Display normalization is trace-wise min-max normalization after smoothing, cropped to 0-2 s, matching the shape-focused overlay view in the previous scripts.",
        "",
        "## Selected Channels",
        "",
    ]
    for dataset, channels in selected.items():
        lines.append(f"- {dataset}: {channels}")
    lines.extend(["", "## Files", ""])
    for p in sorted(out_dir.glob("*_low_high.png")):
        lines.append(f"- `{p.name}`")
    pass


def run(n_select_override: int | None = None) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_selected: dict[str, list[int]] = {}
    ranking_summary: dict[str, list[dict[str, float | int]]] = {}
    metric_rows: list[dict[str, object]] = []

    for spec in SPECS:
        if n_select_override is not None:
            spec = DatasetSpec(**{**spec.__dict__, "n_select": n_select_override})

        spec_out = OUT_DIR / spec.key
        spec_out.mkdir(parents=True, exist_ok=True)

        t, raw_means, mode_means, n_channels = load_day_class_means(spec)
        ranking = rank_channels(spec, t, raw_means, n_channels)
        channels = selected_channels(spec, ranking, n_channels)
        all_selected[spec.title] = channels
        ranking_summary[spec.title] = ranking[:20]
        rank_by_ch = {int(row["channel"]): row for row in ranking}

        for order, ch in enumerate(channels, start=1):
            curves = make_channel_curves(spec, ch, t, raw_means, mode_means)
            draw_channel_figure(spec, curves, spec_out)
            rank_row = rank_by_ch[ch]
            metric_rows.append(
                {
                    "dataset": spec.title,
                    "channel": ch,
                    "selected_channel_order": order,
                    "low_drift_score": rank_row["low_drift_score"],
                    "low_rank_mean_corr": curves["low_mean_corr"],
                    "high_reconstruction_mean_corr": curves["high_mean_corr"],
                    "delta_high_minus_low_corr": curves["delta_corr"],
                    "anchor_dynamic_range": rank_row["anchor_dynamic_range"],
                    "current_channel_from_previous_panel": int(ch in spec.current_channels),
                    "figure_png": str((spec_out / f"{spec.key}_ch{ch:03d}_low_high.png").relative_to(OUT_DIR)),
                }
            )

    write_outputs(OUT_DIR, all_selected, ranking_summary, metric_rows)
    print(f"saved to {OUT_DIR}")
    for dataset, channels in all_selected.items():
        print(f"{dataset}: {channels}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-select", type=int, default=None, help="Override selected channel count per dataset.")
    parser.add_argument('--panel', choices=['A','C','E'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    global SPECS, OUT_DIR
    key = {'A':'ecog_speech','C':'ajile12','E':'shu_eeg'}[args.panel]
    SPECS = [spec for spec in SPECS if spec.key == key]
    if len(SPECS) != 1: raise ValueError('Dataset specification not found: '+key)
    OUT_DIR = args.output
    run(n_select_override=args.n_select)


if __name__ == "__main__":
    main()
