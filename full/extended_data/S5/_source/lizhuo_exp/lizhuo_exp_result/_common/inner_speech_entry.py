

from pathlib import Path

import inner_speech_runtime as runtime


repo = Path(__file__).resolve().parents[3]
runtime.REPO = repo
runtime.CANONICAL_PATH = repo / "data_check_20260507/full_run_5fold/classify_poe_v5_kfold_fp16.py"
runtime.DEFAULT_H5 = repo / "data_check_20260507/full_run_5fold/stage2_new/INNER_class8"
runtime.DEFAULT_MAE = repo / "data_check_20260507/full_run_5fold/ckpts/eeg_mae_INNER_20260508_025549.pt"
runtime.DEFAULT_BRAINOMNI = repo / "weights/BrainOmni/BrainOmni/tiny"


if __name__ == "__main__":
    runtime.main()
