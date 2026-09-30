
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNTIME = REPO / "lizhuo_exp/lizhuo_exp_result/_common/inner_speech_entry.py"
DEFAULT_CHECKPOINT = REPO / (
    "lizhuo_exp/lizhuo_exp_result/Exp_EDFig6a_Bimodal_Inner_Speech_8class/"
    "weights/seed0/fold_00.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh inference from a saved Inner Speech 8-class checkpoint"
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "inference_output")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    command = [
        args.python, str(RUNTIME), "inference",
        "--checkpoint", str(args.checkpoint.expanduser().resolve()),
        "--output-dir", str(args.output_dir.expanduser().resolve()),
        "--split", args.split, "--batch-size", str(args.batch_size),
        "--device", args.device, *extra,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
