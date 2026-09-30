


from __future__ import annotations

import argparse
from pathlib import Path

from poe_eeg_runtime import InferenceRequest, infer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--h5-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--input-h5", type=Path)
    parser.add_argument("--d-path", type=Path)
    parser.add_argument("--lam-cortex-path", type=Path)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--modality", choices=("eeg", "meg"), default="eeg")
    parser.add_argument("--label-key", default="labels")
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--n-classes", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=2560)
    parser.add_argument("--ratio", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "inference_output")
    args = parser.parse_args()
    config = {
        "modality": args.modality,
        "label_key": args.label_key,
        "label_offset": args.label_offset,
        "n_classes": args.n_classes,
        "n_samples": args.n_samples,
        "ratio": args.ratio,
        "d_path": str(args.d_path.resolve()) if args.d_path else None,
        "lam_cortex_path": str(args.lam_cortex_path.resolve()) if args.lam_cortex_path else None,
        "prior_checkpoint": str(args.prior_checkpoint.resolve()) if args.prior_checkpoint else None,
    }
    request = InferenceRequest(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_root,
        h5_dir=args.h5_dir,
        subject=args.subject,
        fold=args.fold,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        input_h5=args.input_h5,
    )
    print(infer(request, config))


if __name__ == "__main__":
    main()
