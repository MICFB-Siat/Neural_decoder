

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]




DATA_ROOT = Path(
    os.environ.get("EIGENMODE_ABLATION_DATA_ROOT", "/tmp/eigenmode_ablation_bfm_data")
).expanduser().resolve()
RESULT_STAGE_ROOT = Path(
    os.environ.get(
        "EIGENMODE_ABLATION_RESULT_STAGE",
        "/tmp/eigenmode_ablation_bfm_result_stage",
    )
).expanduser().resolve()
PROJECT = ROOT.parents[2]
STAGE2 = PROJECT / "data_check_20260507/full_run_5fold/stage2_new"
EIGENMODE = PROJECT / "guoyi_exp/data/eigenmode_test"

REQUESTED_KS = (1, 3, 5, 10, 20, 40, 48)
SEEDS = tuple(range(5))



CONFIRMED_KS = {
    "BCIC": (1, 3, 5, 10, 17),
    "FACED": (1, 3, 5, 10, 20, 25),
    "MOTOR": REQUESTED_KS,
}
ALL_KS = tuple(sorted({k for values in CONFIRMED_KS.values() for k in values}))

DATASETS = {
    "BCIC": {
        "source": STAGE2 / "BCIC",
        "template": EIGENMODE / "BCIC/d_template_eeg_17_5e253942b967.npy",
        "label_key": "labels",
        "n_classes": 4,
    },
    "FACED": {
        "source": STAGE2 / "FACED_new",
        "template": EIGENMODE / "FACED/d_template_eeg_25.npy",
        "label_key": "label2",
        "n_classes": 3,
    },
    "MOTOR": {
        "source": STAGE2 / "MOTOR",
        "template": EIGENMODE / "MOTOR_eeg-fmri/D_eeg.npy",
        "label_key": "labels",
        "n_classes": 2,
    },
}

CLASSIFIER = ROOT.parent / "Exp_Classification/code/classify_baseline_v2.py"
CHECKPOINT = PROJECT / "checkpoints/unified_v3/prior_unified_v3_20260514_020311.pt"
FS32K = EIGENMODE / "fs32k"
