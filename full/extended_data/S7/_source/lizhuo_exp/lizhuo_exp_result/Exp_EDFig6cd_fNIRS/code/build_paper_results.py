


from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "result/source_data/edfig6cd_seed0_per_fold_metrics.json"
PAPER_N = {"Dubois2024": 49, "Duwadi2026": 30}
DATASET_LABEL = {"dubois": "Dubois2024", "cocktail": "Duwadi2026"}
METHOD_FIELD = {
    "fNIRS Transformer": "fnirs_transformer",
    "Ours": "ours_poe_val_weighted",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "inference_output")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    fold_rows = []
    for row in records:
        dataset = DATASET_LABEL[row["dataset"]]
        for method, field in METHOD_FIELD.items():
            value = row.get(field, {})
            fold_rows.append({
                "panel": "ED Fig. 6c" if dataset == "Dubois2024" else "ED Fig. 6d",
                "dataset": dataset,
                "subject": row["subject"],
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "method": method,
                "accuracy": value.get("acc"),
                "balanced_accuracy": value.get("balacc"),
            })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "edfig6cd_seed0_fold_accuracy.csv", index=False)
    subjects = (folds.groupby(["panel", "dataset", "subject", "seed", "method"], as_index=False)
                .agg(mean_accuracy=("accuracy", "mean"),
                     std_accuracy=("accuracy", "std"),
                     mean_balanced_accuracy=("balanced_accuracy", "mean"),
                     n_folds=("fold", "nunique")))
    subjects.to_csv(out / "edfig6cd_subject_accuracy.csv", index=False)

    coverage = []
    for dataset, paper_n in PAPER_N.items():
        panel = "ED Fig. 6c" if dataset == "Dubois2024" else "ED Fig. 6d"
        for method in METHOD_FIELD:
            part = subjects[(subjects.dataset == dataset) & (subjects.method == method)]
            observed = int(part.subject.nunique())
            folds_per_subject = sorted({int(x) for x in part.n_folds})
            coverage.append({
                "panel": panel,
                "dataset": dataset,
                "method": method,
                "paper_n": paper_n,
                "exported_n": observed,
                "complete": observed == paper_n and folds_per_subject == [5],
                "folds_per_subject": "|".join(map(str, folds_per_subject)),
                "seed": 0,
                "evidence_status": "completed_joint_seed0_result",
            })
    pd.DataFrame(coverage).to_csv(out / "edfig6cd_method_coverage.csv", index=False)

    manifest = {
        "status": "complete_demonstration_seed0",
        "paper_subject_contract": PAPER_N,
        "method_field_contract": METHOD_FIELD,
        "coverage": coverage
    }
    (out / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
