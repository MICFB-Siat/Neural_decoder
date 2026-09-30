




from __future__ import annotations

import argparse
import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import functional as F

from models import EEGMEGPriorEncoder, PoEFusion, QFormer, SubjectLayers
from nsd_reconstruction import DEFAULT_DATA as DEFAULT_NSD_DATA
from nsd_reconstruction import DEFAULT_OUTPUT as DEFAULT_NSD_OUTPUT
from nsd_reconstruction import run_nsd_reconstruction


ROOT = Path(__file__).resolve().parent


def _display_path(path: Path) -> str:

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _strings(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(pearsonr(a, b).statistic)


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    return None if denom == 0 else float(np.dot(a.ravel(), b.ravel()) / denom)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for cls in np.unique(np.concatenate([labels, predictions])):
        tp = np.sum((labels == cls) & (predictions == cls))
        fp = np.sum((labels != cls) & (predictions == cls))
        fn = np.sum((labels == cls) & (predictions != cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def run_classification(h5: h5py.File, weights: Path, device: torch.device, output: Path) -> dict:
    data = np.asarray(h5["classification/eeg_modes"], dtype=np.float32)
    labels = np.asarray(h5["classification/label"], dtype=np.int64)
    source_index = np.asarray(h5["classification/source_index"], dtype=np.int64)
    checkpoint = torch.load(weights / "classification_stage2_eeg_tiny.pth", map_location="cpu",
                            weights_only=True)
    bci = checkpoint["bci"]
    encoder = EEGMEGPriorEncoder().to(device)
    encoder.load_state_dict(bci["eegmeg_prior"], strict=True)
    head = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(0.0), nn.Linear(1024, 5)).to(device)
    head.load_state_dict(bci["head_prior"], strict=True)
    encoder.eval(); head.eval()
    with torch.inference_mode():
        tokens = encoder(torch.from_numpy(data).to(device))
        logits = head(tokens.mean(1))
        probabilities = logits.softmax(-1).cpu().numpy()
    predictions = probabilities.argmax(1)
    rows = [{"sample": int(i), "source_index": int(src), "true_class": int(y),
             "predicted_class": int(p), "confidence": float(probabilities[i, p])}
            for i, (src, y, p) in enumerate(zip(source_index, labels, predictions))]
    _write_csv(output / "classification_predictions.csv", rows)
    return {"n": int(len(labels)), "accuracy": float(np.mean(labels == predictions)),
            "macro_f1": _macro_f1(labels, predictions),
            "weights": _display_path(weights / "classification_stage2_eeg_tiny.pth")}


def _load_language_alignment(weights: Path, device: torch.device):
    checkpoint = torch.load(weights / "alice_qformer_poe_sublayer.pt", map_location="cpu",
                            weights_only=True)
    config = checkpoint["args"]
    subject_ids = list(checkpoint["subject_ids"])
    fusion = PoEFusion().to(device)
    fusion.load_state_dict(checkpoint["poe"], strict=True)
    layers = SubjectLayers(len(subject_ids), 512).to(device)
    layers.load_state_dict(checkpoint["sub_layers"], strict=True)
    qformer = QFormer(n_queries=20, d_q=int(config["qf_dq"]), d_in=512, d_out=1024,
                      n_layers=int(config["qf_layers"]), n_heads=int(config["qf_heads"])).to(device)
    qformer.load_state_dict(checkpoint["qformer"], strict=True)
    fusion.eval(); layers.eval(); qformer.eval()
    return fusion, layers, qformer, subject_ids


def language_alignment(h5: h5py.File, weights: Path, device: torch.device, output: Path):
    obs = np.asarray(h5["language/bci_obs"], dtype=np.float32)
    prior = np.asarray(h5["language/bci_prior"], dtype=np.float32)
    target = np.asarray(h5["language/bge_target"], dtype=np.float32)
    text = _strings(h5["language/text"][...])
    subject_index = np.asarray(h5["language/subject_index"], dtype=np.int64)
    source_index = np.asarray(h5["language/source_index"], dtype=np.int64)
    fusion, layers, qformer, subject_ids = _load_language_alignment(weights, device)
    with torch.inference_mode():
        sid = torch.from_numpy(subject_index).long().to(device)
        aligned = qformer(layers(fusion(torch.from_numpy(obs).to(device),
                                         torch.from_numpy(prior).to(device)), sid)).cpu().numpy()
    rows = []
    for i, (prediction, truth) in enumerate(zip(aligned, target)):
        rows.append({"sample": i, "source_index": int(source_index[i]),
                     "subject": subject_ids[int(subject_index[i])], "ground_truth": text[i],
                     "bge_token_cosine": _cosine(prediction, truth),
                     "bge_token_pearson_r": _pearson(prediction, truth)})
    _write_csv(output / "language_alignment.csv", rows)
    return aligned, text, rows, {"n": len(rows), "mean_bge_token_cosine": float(np.mean([r["bge_token_cosine"] for r in rows])),
                                 "mean_bge_token_pearson_r": float(np.mean([r["bge_token_pearson_r"] for r in rows])),
                                 "subject_ids_in_checkpoint": subject_ids}


def _polish_english(text: str, max_chars=500) -> str:
    import re
    text = re.sub(r"[^0-9A-Za-z ,.!?'À-ɏ-]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()[:max_chars]


def _bge_sentence_embeddings(texts: list[str], model_path: str, device: torch.device) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        result = F.normalize(pooled, dim=-1).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def language_text_decode(aligned: np.ndarray, references: list[str], weights: Path,
                         phi_path: str, bge_path: str, device: torch.device, output: Path,
                         max_new_tokens: int) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.utils as transformer_utils
    if not hasattr(transformer_utils, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            pass
        transformer_utils.LossKwargs = LossKwargs

    trace_path = output / "language_text_trace.txt"
    def trace(stage: str) -> None:
        trace_path.write_text(stage + "\n", encoding="utf-8")

    trace("start")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(phi_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        phi_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    llm = PeftModel.from_pretrained(llm, weights / "alice_lora").eval()
    trace("phi_and_lora_loaded")
    hidden_size = int(llm.config.hidden_size)
    projector = nn.Sequential(nn.Linear(1024, hidden_size), nn.LayerNorm(hidden_size)).to(device)
    projector.load_state_dict(torch.load(weights / "alice_projector.pt", map_location="cpu", weights_only=True), strict=True)
    projector.eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a neural signal decoder. Based on the fMRI brain activity signal, output the English text the subject was listening to."},
         {"role": "user", "content": "Decode the text corresponding to the following brain signal:\n<__BCI__>"}],
        tokenize=False, add_generation_prompt=True)
    prefix_text, suffix_text = prompt.split("<__BCI__>", 1)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.inference_mode():
        prefix = llm.get_input_embeddings()(prefix_ids).to(dtype)
        suffix = llm.get_input_embeddings()(suffix_ids).to(dtype)
        soft = projector(torch.from_numpy(aligned).to(device)).to(dtype)
        inputs = torch.cat([prefix.expand(len(aligned), -1, -1), soft,
                            suffix.expand(len(aligned), -1, -1)], dim=1)
        mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
        generated = llm.generate(inputs_embeds=inputs, attention_mask=mask,
                                 max_new_tokens=max_new_tokens, num_beams=3,
                                 early_stopping=True, repetition_penalty=1.3,
                                 no_repeat_ngram_size=4, pad_token_id=tokenizer.eos_token_id)
    trace("generation_finished")
    try:
        decoded = [_polish_english(tokenizer.decode(row, skip_special_tokens=True))
                   for row in generated.detach().cpu().tolist()]
    except Exception:
        trace("decode_failed\n" + traceback.format_exc())
        raise
    _write_csv(output / "language_text_predictions.csv",
               [{"sample": i, "ground_truth": reference, "decoded_text": hypothesis}
                for i, (reference, hypothesis) in enumerate(zip(references, decoded))])
    trace("text_csv_written")
    del llm, projector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reference_bge = _bge_sentence_embeddings(references, bge_path, device)
    trace("reference_bge_finished")
    decoded_bge = _bge_sentence_embeddings(decoded, bge_path, device)
    trace("decoded_bge_finished")
    score = np.sum(reference_bge * decoded_bge, axis=1)
    rows = [{"sample": i, "ground_truth": reference, "decoded_text": hypothesis,
             "text_bge_cosine": float(score[i])}
            for i, (reference, hypothesis) in enumerate(zip(references, decoded))]
    _write_csv(output / "language_text_predictions.csv", rows)
    return {"n": len(rows), "mean_text_bge_cosine": float(score.mean()),
            "phi_path": phi_path, "bge_path": bge_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen multimodal brain-decoding inference")
    parser.add_argument("--task", choices=["classification", "language-align", "language", "image", "all"], default="all")
    parser.add_argument("--data", type=Path, default=ROOT / "assets/inference_data.h5")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-data", type=Path, default=DEFAULT_NSD_DATA)
    parser.add_argument("--image-output", type=Path, default=DEFAULT_NSD_OUTPUT)
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-count", type=int, default=10,
                        help="Consecutive Shared1000 images; 0 means all remaining images.")
    parser.add_argument("--image-prior-steps", type=int, default=20)
    parser.add_argument("--image-reconstruction-steps", type=int, default=50)
    parser.add_argument("--phi-path", help="Local Phi-4-mini-instruct base model; required for --task language")
    parser.add_argument("--bge-path", help="Local BGE-M3 model; required for --task language")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--language-n", type=int, default=None,
                        help="Decode only the first N fixed language examples (useful for a fast smoke test).")
    args = parser.parse_args()
    if args.task == "language" and (not args.phi_path or not args.bge_path):
        parser.error("--task language requires both --phi-path and --bge-path")
    if args.task != "image" and not args.data.is_file():
        raise FileNotFoundError(f"Inference data not found: {args.data}. Run prepare_assets.py first.")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
               "task": args.task}
    if args.task != "image":
      with h5py.File(args.data, "r") as h5:
        aligned = references = None
        if args.task in {"classification", "all"}:
            summary["classification"] = run_classification(h5, args.weights, device, args.output)
        if args.task in {"language-align", "language", "all"}:
            aligned, references, _, result = language_alignment(h5, args.weights, device, args.output)
            summary["language_alignment"] = result
        if args.task == "language":
            if args.language_n is not None:
                if args.language_n < 1:
                    parser.error("--language-n must be at least 1")
                aligned, references = aligned[:args.language_n], references[:args.language_n]
            summary["language_text"] = language_text_decode(aligned, references, args.weights,
                                                              args.phi_path, args.bge_path, device,
                                                              args.output, args.max_new_tokens)
    if args.task in {"image", "all"}:
        summary["image_reconstruction"] = run_nsd_reconstruction(
            data_path=args.image_data,
            output=args.image_output,
            device_name=args.device,
            start=args.image_start,
            count=args.image_count,
            prior_steps=args.image_prior_steps,
            reconstruction_steps=args.image_reconstruction_steps,
        )
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()
