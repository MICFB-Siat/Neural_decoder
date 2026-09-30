
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from resources import path as resource_path

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler


AJILE_BANDS = ((4, 8), (8, 13), (13, 30), (30, 70), (70, 150))
SHU_BANDS = ((4, 8), (8, 13), (13, 30), (30, 50))


@dataclass
class Result:
    dataset: str
    labels: list[str]
    days: list[int]
    low: np.ndarray
    high: np.ndarray


def flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(values.shape[0], -1)


def bandpower(values: np.ndarray, frequency: float, bands) -> np.ndarray:
    frequencies, spectrum = welch(values, fs=frequency,
                                   nperseg=min(256, values.shape[-1]), axis=-1)
    features = []
    for lower, upper in bands:
        mask = (frequencies >= lower) & (frequencies < upper)
        features.append(np.log(np.abs(spectrum[:, :, mask].mean(axis=-1)) + 1e-12))
    return np.concatenate(features, axis=1)


def session_standardize(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    output = values.astype(np.float64, copy=True)
    for session in np.unique(sessions):
        mask = sessions == session
        output[mask] = StandardScaler().fit_transform(values[mask])
    return output


def displacement(values: np.ndarray, labels: np.ndarray, sessions: np.ndarray, days: list[int]) -> np.ndarray:
    values = StandardScaler().fit_transform(flatten(values))
    classes = sorted(map(int, np.unique(labels)))
    for day in days:
        if any(not np.any((sessions == day) & (labels == item)) for item in classes):
            raise ValueError(f"Missing class in session {day}")
    reference = np.asarray([values[(sessions == days[0]) & (labels == item)].mean(axis=0) for item in classes])
    scale = np.mean(np.linalg.norm(reference - values[sessions == days[0]].mean(axis=0), axis=1))
    reference_vector = reference.reshape(-1)
    result = []
    for day in days:
        centroids = np.asarray([values[(sessions == day) & (labels == item)].mean(axis=0) for item in classes])
        result.append(np.linalg.norm(centroids.reshape(-1) - reference_vector) / (scale + 1e-12))
    return np.asarray(result)


def load_ecog(root: Path) -> Result:
    labels = np.load(root / "labels.npy").astype(int)
    sessions = np.load(root / "session.npy").astype(int)
    low = flatten(np.load(root / "F_multiband.npy"))
    high = flatten(np.load(root / "feat_z.npy"))
    days = [0, 6, 8, 13, 14, 18, 35, 42, 43, 204, 208]
    return Result("ECoG Speech", [f"D{day}" for day in days], days, displacement(low, labels, sessions, days), displacement(high, labels, sessions, days))


def load_h5(path: Path, dataset: str, representation: Path | None = None) -> Result:
    with h5py.File(path, "r") as handle:
        raw = handle["eeg_raw"][:]
        labels = handle["labels"][:].astype(int)
        sessions = handle["session"][:].astype(int)
        frequency = float(handle.attrs.get("eeg_sr", 256.0))
        if dataset == "AJILE12":
            high = session_standardize(
                bandpower(handle["eeg_modes"][:], frequency, AJILE_BANDS), sessions)
    bands = AJILE_BANDS if dataset == "AJILE12" else SHU_BANDS
    low = bandpower(raw, frequency, bands)
    if dataset == "SHU EEG":
        with np.load(representation) as cache:
            if not np.array_equal(labels, cache["y"]) or not np.array_equal(sessions, cache["ses"]):
                raise ValueError("SHU raw data and model representation are not trial-aligned")
            high = cache["ours"]
    days = sorted(map(int, np.unique(sessions)))
    return Result(dataset, [f"S{day}" for day in days], days,
                  displacement(low, labels, sessions, days),
                  displacement(high, labels, sessions, days))


def save_source(results: list[Result], output: Path) -> None:
    rows = []
    for result in results:
        for label, day, low, high in zip(result.labels, result.days, result.low, result.high):
            rows.append({"dataset": result.dataset, "session": label, "session_value": day, "low_level": float(low), "high_level": float(high)})
    with (output / "Fig4JK_all_sessions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


def draw(results: list[Result], output: Path) -> None:
    ecog = results[0]
    shown = [ecog.days.index(day) for day in (8, 14, 42, 43, 204, 208)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    ax.axvspan(3.55, 5.3, color="#E5DDCE", zorder=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=.8)
    ax.plot(np.arange(len(shown)), ecog.low[shown], "o-", color="#9A9B9E", label="Low-level")
    ax.plot(np.arange(len(shown)), ecog.high[shown], "o-", color="#2D61A4", label="High-level")
    ax.set_xticks(np.arange(len(shown)), [f"D{ecog.days[index]}" for index in shown])
    ax.set_ylabel("Representational drift"); ax.set_xlabel("days"); ax.legend(frameon=False)
    ax.set_yticks(np.arange(0, 20, 2.5))
    ax.set_ylim(-1, 20)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(output / "Fig4J.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "Fig4J.pdf", bbox_inches="tight")
    fig.savefig(output / "Fig4J.svg", bbox_inches="tight")
    plt.close(fig)

    selected = [(results[0], 208), (results[1], 7), (results[2], 4)]
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    for row, (result, day) in enumerate(selected[::-1]):
        index = result.days.index(day)
        low, high = float(result.low[index]), float(result.high[index])
        ax.plot([high, low], [row, row], color="#D6D8DA", lw=3)
        ax.scatter(low, row, color="#A4A6A9", s=55, zorder=3)
        ax.scatter(high, row, color="#2D61A4", s=55, zorder=3)
        ax.text(low, row + .16, f"{low:.1f}", ha="center", color="#888888", fontsize=7)
        ax.text(high, row + .16, f"{high:.1f}", ha="center", color="#2D61A4", fontsize=7)
        ax.text((low+high)/2, row + .20, f"{low/high:.1f} x", ha="center", color="#D45454", fontsize=7)
    ax.set_yticks(range(3), [item[0].dataset for item in selected[::-1]])
    ax.set_ylim(-.4, 2.5)
    ax.set_xlim(0, 20)
    ax.scatter([], [], color="#A4A6A9", s=30, label="Low-level")
    ax.scatter([], [], color="#2D61A4", s=30, label="High-level")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.set_xlabel("Representational drift from reference session")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(output / "Fig4K.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "Fig4K.pdf", bbox_inches="tight")
    fig.savefig(output / "Fig4K.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecog", type=Path, default=resource_path('ecog_drift'))
    parser.add_argument("--ajile", type=Path, default=resource_path('ajile_h5'))
    parser.add_argument("--shu", type=Path, default=resource_path('shu_h5'))
    parser.add_argument("--shu-representation", type=Path, default=resource_path('shu_repr'))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "fig4jk_paper")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    results = [load_ecog(args.ecog), load_h5(args.ajile, "AJILE12"), load_h5(args.shu, "SHU EEG", args.shu_representation)]
    save_source(results, args.output); draw(results, args.output)
    j = results[0]
    j_rows = [{"days_since_D0": d, "low_level": float(j.low[j.days.index(d)]),
               "high_level": float(j.high[j.days.index(d)])} for d in (8,14,42,43,204,208)]
    k_rows = []
    for result, day in zip(results, (208, 7, 4)):
        i = result.days.index(day)
        k_rows.append({"dataset": result.dataset, "session": day,
                       "low_level": float(result.low[i]), "high_level": float(result.high[i])})
    for name, rows in (("Fig4J_source.csv", j_rows), ("Fig4K_source.csv", k_rows)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "processing.json").write_text(json.dumps({
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "output"},
        "metric": "L2 displacement of concatenated class centroids divided by reference mean class-to-population-centroid distance",
        "standardization": "Global per-feature StandardScaler fitted to all retained trials, including follow-up sessions; descriptive analysis",
        "trials": "All retained trials per session; no trial selection by output",
        "ecog": "F_multiband versus retained feat_z; feat_z already session-aligned",
        "ajile": "Welch log bandpower: raw versus per-session-standardized eigenmodes",
        "shu": "Welch log raw bandpower versus saved ours model features; no additional session alignment",
        "K_sessions": {"ECoG Speech": 208, "AJILE12": 7, "SHU EEG": 4},
        "training_performed": False}, indent=2))
    print(json.dumps({"J": j_rows, "K": k_rows}, indent=2))


if __name__ == "__main__":
    main()
