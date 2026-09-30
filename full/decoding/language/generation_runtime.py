


from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from model import PLACEHOLDER, PoEFusion, QFormer, SubjectLayers, decode_batch, polish_text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_torch_load(path: Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def text(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8", errors="replace") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in values]


def weight_files(weight_dir: Path) -> dict[str, Path]:
    result = {
        "stage_a": weight_dir / "qformer_poe_sublayer.pt",
        "projector": weight_dir / "projector.pt",
        "lora": weight_dir / "lora/adapter_model.safetensors",
        "lora_config": weight_dir / "lora/adapter_config.json",
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing generation weight(s): " + ", ".join(missing))
    return result


def architecture(stage: dict) -> dict[str, int]:
    q = stage["qformer"]
    block_ids = {int(k.split(".")[1]) for k in q if k.startswith("blocks.")}
    return {
        "n_subjects": int(stage["sub_layers"]["weights"].shape[0]),
        "d_in": int(q["input_proj.weight"].shape[1]),
        "d_q": int(q["queries"].shape[2]),
        "n_queries": int(q["queries"].shape[1]),
        "d_out": int(q["out_proj.weight"].shape[0]),
        "n_layers": max(block_ids) + 1,
        "n_heads": 8,
    }


def check_contract(weight_dir: Path, data_dir: Path) -> dict:
    files = weight_files(weight_dir)
    stage = safe_torch_load(files["stage_a"])
    subjects = list(stage["subject_ids"])
    missing = [str(data_dir / f"{s}.h5") for s in subjects if not (data_dir / f"{s}.h5").is_file()]
    if missing:
        raise FileNotFoundError("missing subject H5(s): " + ", ".join(missing[:10]))
    info = architecture(stage)
    info.update({"subjects": subjects, "fusion_mode": stage.get("fusion_mode", "poe"),
                 "weight_dir": str(weight_dir), "data_dir": str(data_dir)})
    return info


def run_generation(*, weight_dir: Path, data_dir: Path, output_dir: Path,
                   lang: str, device_name: str,
                   batch_size: int, beam: int, max_new: int,
                   polish_max_chars: int, prior_key: str = "bci_prior",
                   subjects: list[str] | None = None, seed: int = 42,
                   llm_path: Path = Path("frozen_models/Phi-4-mini-instruct"),
                   check_only: bool = False, limit: int | None = None) -> dict:
    weight_dir, data_dir, output_dir = weight_dir.resolve(), data_dir.resolve(), output_dir.resolve()
    files = weight_files(weight_dir)
    stage = safe_torch_load(files["stage_a"])
    arch = architecture(stage)
    checkpoint_subjects = list(stage["subject_ids"])
    selected = checkpoint_subjects if subjects is None else subjects
    unknown = sorted(set(selected) - set(checkpoint_subjects))
    if unknown:
        raise ValueError(f"subjects absent from checkpoint: {unknown}")
    for subject in selected:
        if not (data_dir / f"{subject}.h5").is_file():
            raise FileNotFoundError(data_dir / f"{subject}.h5")
    contract = {**arch, "subjects_in_checkpoint": checkpoint_subjects, "selected_subjects": selected,
                "fusion_mode": stage.get("fusion_mode", "poe")}
    if check_only:
        print(json.dumps(contract, indent=2))
        return contract

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    sub_layers = SubjectLayers(arch["n_subjects"], arch["d_in"], init_id=True).to(device)
    qformer = QFormer(arch["n_queries"], arch["d_q"], arch["d_in"], arch["d_out"],
                      arch["n_layers"], arch["n_heads"]).to(device)
    sub_layers.load_state_dict(stage["sub_layers"], strict=True)
    qformer.load_state_dict(stage["qformer"], strict=True)
    poe = None
    if stage.get("fusion_mode", "poe") == "poe":
        poe = PoEFusion().to(device)
        poe.load_state_dict(stage["poe"], strict=True)
        poe.eval()
    sub_layers.eval(); qformer.eval()


    aligned: dict[str, torch.Tensor] = {}
    ground_truth: dict[str, list[str]] = {}
    sample_indices: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for subject in selected:
            subject_position = checkpoint_subjects.index(subject)
            with h5py.File(data_dir / f"{subject}.h5", "r") as f:
                obs = f["bci_obs"][:].astype(np.float32)
                prior = f[prior_key][:].astype(np.float32)
                original_indices = f["original_sample_index"][:] if "original_sample_index" in f else np.arange(len(obs))
                labels = text(f["labels"][:]) if "labels" in f else [""] * len(obs)
            idx = np.arange(len(obs))
            idx = idx[:limit] if limit is not None else idx
            sample_indices[subject] = original_indices[idx]
            ground_truth[subject] = [labels[int(i)] for i in idx]
            chunks = []
            for start in range(0, len(idx), batch_size):
                take = idx[start:start + batch_size]
                a = torch.from_numpy(obs[take]).to(device)
                b = torch.from_numpy(prior[take]).to(device)
                fused = poe(a, b) if poe is not None else a
                sid = torch.full((len(take),), subject_position, dtype=torch.long, device=device)
                chunks.append(qformer(sub_layers(fused, sid)).cpu())
            aligned[subject] = torch.cat(chunks)
    del stage, sub_layers, qformer, poe
    if device.type == "cuda": torch.cuda.empty_cache()

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16).to(device)
    if lang == "DE":
        system = "Du bist ein neuronaler Signaldecoder. Gib basierend auf dem fMRT-Hirnsignal den entsprechenden deutschen Text aus."
        user = "Dekodiere den Text, der dem folgenden Hirnsignal entspricht:"
    else:
        language = {"CN": "Chinese", "EN": "English", "FR": "French"}[lang]
        system = f"You are a neural signal decoder. Based on the fMRI brain activity signal, output the {language} text the subject was listening to."
        user = "Decode the text corresponding to the following fMRI signal:"
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": f"{user}\n{PLACEHOLDER}"}],
        tokenize=False, add_generation_prompt=True,
    )
    before, after = prompt.split(PLACEHOLDER, 1)
    before_ids = tokenizer(before, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    after_ids = tokenizer(after, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.inference_mode():
        prefix = llm.get_input_embeddings()(before_ids).to(torch.bfloat16)
        suffix = llm.get_input_embeddings()(after_ids).to(torch.bfloat16)
    config = json.loads(files["lora_config"].read_text(encoding="utf-8"))
    lora = LoraConfig(r=int(config["r"]), lora_alpha=int(config["lora_alpha"]),
                      target_modules=config["target_modules"],
                      lora_dropout=float(config.get("lora_dropout", 0.0)), bias="none",
                      init_lora_weights="pissa")
    llm = get_peft_model(llm, lora)
    trained = load_file(str(files["lora"]))
    expected = set(get_peft_model_state_dict(llm))
    matched = len(expected & set(trained))
    result = set_peft_model_state_dict(llm, trained)
    if matched != len(expected) or matched == 0:
        raise RuntimeError(f"LoRA key mismatch: matched {matched}/{len(expected)}")
    llm.eval()
    llm.requires_grad_(False)
    projector = nn.Sequential(nn.Linear(arch["d_out"], llm.config.hidden_size),
                              nn.LayerNorm(llm.config.hidden_size)).to(device)
    projector.load_state_dict(safe_torch_load(files["projector"], device), strict=True)
    projector.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in selected:
        predictions = []
        values = aligned[subject]
        print(f"Generating {subject}: {len(values)} trials", flush=True)
        for start in range(0, len(values), batch_size):
            decoded = decode_batch(llm, projector, tokenizer, prefix, suffix,
                                   values[start:start + batch_size], device, beam, max_new)
            predictions.extend(decoded)
            print(f"{subject}: {len(predictions)}/{len(values)} generated", flush=True)
        output = [polish_text(value, lang, polish_max_chars) for value in predictions]
        for index, gt, raw, clean in zip(sample_indices[subject], ground_truth[subject], predictions, output):
            rows.append({"subject": subject, "sample_index": int(index), "ground_truth": gt,
                         "decoded": raw, "output": clean})
        subject_dir = output_dir / subject
        subject_dir.mkdir(exist_ok=True)
        with (subject_dir / "test_decode.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[-len(predictions)])); writer.writeheader()
            writer.writerows(rows[-len(predictions):])
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {
        "fresh_model_forward": True, "language": lang, "n_predictions": len(rows),
        "sample_selection": "input_samples", "limit_per_subject": limit,
        "architecture": arch, "subjects": selected, "device": str(device),
        "lora_keys_matched": matched, "lora_keys_expected": len(expected),
        "weights_are_external_pointers": False,
        "weights": [{"role": role, "path": str(path), "sha256": sha256(path)}
                    for role, path in files.items() if role != "lora_config"],
        "data_dir": str(data_dir), "output_dir": str(output_dir),
    }
    (output_dir / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest
