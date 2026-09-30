













































from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


__all__ = [
    "calculate_bge_score_f1",
    "BGEScoreF1Evaluator",
    "ScoreResult",
]



_LOCAL_BGE_M3 = (
    str(_LOCAL_SOURCE / 'jch_exp/10_bayesian_decode_smn/cache/bge-m3')
)
_DEFAULT_HF_ID = "BAAI/bge-m3"
_SUPPORTED_LANGS = {"en", "zh", "multi"}


class ScoreResult:








    def __init__(self, P, R, F1):
        self.precision = P
        self.recall = R
        self.f1 = F1

    @property
    def mean(self) -> Tuple[float, float, float]:
        return (
            float(self.precision.mean().item()),
            float(self.recall.mean().item()),
            float(self.f1.mean().item()),
        )

    def as_dict(self) -> dict:
        mp, mr, mf = self.mean
        return {
            "precision_mean": mp,
            "recall_mean": mr,
            "f1_mean": mf,
            "precision_per_sample": [float(x) for x in self.precision.tolist()],
            "recall_per_sample": [float(x) for x in self.recall.tolist()],
            "f1_per_sample": [float(x) for x in self.f1.tolist()],
        }


def _normalize_lang(lang: str) -> str:

    if lang is None:
        return "multi"
    l = lang.strip().lower()
    if l in {"en", "english", "eng"}:
        return "en"
    if l in {"zh", "zh_cn", "zh-cn", "chinese", "cn"}:
        return "zh"
    if l in {"multi", "multilingual", "mix", "ml"}:
        return "multi"
    raise ValueError(
        f"Unsupported lang={lang!r}. Supported: {_SUPPORTED_LANGS}."
    )


def _coerce_text_list(x) -> List[str]:
    if isinstance(x, (tuple, list)):
        return [str(s) for s in x]
    raise TypeError(f"Expected list/tuple of strings, got {type(x).__name__}.")


def _resolve_model_path(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if os.path.isdir(_LOCAL_BGE_M3):
        return _LOCAL_BGE_M3
    return _DEFAULT_HF_ID


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEScoreF1Evaluator:





    def __init__(
        self,
        model_path: Optional[str] = None,
        lang: str = "multi",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = True,
        strip_special: bool = False,
        verbose: bool = False,
    ):
        self.lang = _normalize_lang(lang)
        self.model_path = _resolve_model_path(model_path)
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.strip_special = strip_special
        self.verbose = verbose

        dtype = (
            torch.float16
            if (use_fp16 and "cuda" in str(self.device))
            else torch.float32
        )
        self.use_fp16 = (dtype == torch.float16)

        if self.verbose:
            print(f"[BGEScoreF1] loading {self.model_path} on {self.device} "
                  f"(fp16={self.use_fp16})")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        try:
            self.model = AutoModel.from_pretrained(self.model_path, dtype=dtype)
        except TypeError:
            self.model = AutoModel.from_pretrained(
                self.model_path, torch_dtype=dtype
            )
        self.model = self.model.to(self.device).eval()

        self._special_ids = torch.tensor(
            list(self.tokenizer.all_special_ids), device=self.device
        )

    @torch.no_grad()
    def _encode_tokens(self, texts: Sequence[str]) -> List[np.ndarray]:

        results: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            h = out.last_hidden_state
            h = F.normalize(h.float(), p=2, dim=-1)

            attn = enc.attention_mask.bool()
            input_ids = enc.input_ids
            if self.strip_special:
                spec_mask = torch.isin(input_ids, self._special_ids)
                token_mask = attn & ~spec_mask
            else:
                token_mask = attn

            for b in range(h.size(0)):
                m = token_mask[b]
                if m.sum().item() == 0:

                    vec = h[b].mean(dim=0, keepdim=True)
                    vec = F.normalize(vec, p=2, dim=-1)
                    results.append(vec.cpu().numpy())
                else:
                    results.append(h[b][m].cpu().numpy())
        return results

    @staticmethod
    def _pair_prf1(
        cand_vecs: np.ndarray, ref_vecs: np.ndarray
    ) -> Tuple[float, float, float]:

        sim = cand_vecs @ ref_vecs.T
        p = float(sim.max(axis=1).mean())
        r = float(sim.max(axis=0).mean())
        f1 = (2 * p * r / (p + r)) if (p + r) > 1e-12 else 0.0
        return p, r, f1

    def score(
        self, cands: Sequence[str], refs: Sequence[str]
    ) -> Tuple[float, float, float]:

        return self.score_detailed(cands, refs).mean

    def score_detailed(
        self, cands: Sequence[str], refs: Sequence[str]
    ) -> ScoreResult:
        cands = _coerce_text_list(cands)
        refs = _coerce_text_list(refs)
        if len(cands) != len(refs):
            raise ValueError(
                f"len(cands)={len(cands)} != len(refs)={len(refs)}."
            )
        if len(cands) == 0:
            raise ValueError("Empty input: no candidate/reference pairs.")

        cand_vecs = self._encode_tokens(cands)
        ref_vecs = self._encode_tokens(refs)

        Ps, Rs, F1s = [], [], []
        for vc, vr in zip(cand_vecs, ref_vecs):
            p, r, f1 = self._pair_prf1(vc, vr)
            Ps.append(p)
            Rs.append(r)
            F1s.append(f1)

        return ScoreResult(
            torch.tensor(Ps, dtype=torch.float32),
            torch.tensor(Rs, dtype=torch.float32),
            torch.tensor(F1s, dtype=torch.float32),
        )


def calculate_bge_score_f1(
    reconstructed_texts: Union[Sequence[str], Iterable[str]],
    original_texts: Union[Sequence[str], Iterable[str]],
    lang: str = "multi",
    model_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    use_fp16: bool = True,
    strip_special: bool = False,
    verbose: bool = False,
) -> Tuple[float, float, float]:




















    reconstructed_texts = _coerce_text_list(list(reconstructed_texts))
    original_texts = _coerce_text_list(list(original_texts))

    ev = BGEScoreF1Evaluator(
        model_path=model_path,
        lang=lang,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
        strip_special=strip_special,
        verbose=verbose,
    )
    return ev.score(reconstructed_texts, original_texts)




def _load_texts(path: str) -> List[str]:

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    ext = p.suffix.lower()
    if ext == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(s) for s in obj]
        if isinstance(obj, dict):
            for key in ("texts", "preds", "predictions", "refs", "references"):
                if key in obj and isinstance(obj[key], list):
                    return [str(s) for s in obj[key]]
        raise ValueError(
            f"Unrecognized JSON layout in {path}; expected list or dict with "
            "key 'texts'/'preds'/'refs'."
        )
    with open(p, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute BGEScore-F1 (BERTScore formula + BGE-m3 backbone) "
                    "between predicted and reference texts."
    )
    ap.add_argument("--pred", required=True, help="预测文本文件 (.txt / .json)")
    ap.add_argument("--ref", required=True, help="参考文本文件 (.txt / .json)")
    ap.add_argument("--lang", default="multi",
                    choices=["en", "zh", "multi", "EN", "ZH"],
                    help="语言标签（仅做记录，BGE-m3 是多语模型）")
    ap.add_argument("--model-path", default=None,
                    help=f"BGE-m3 权重目录；缺省优先用 {_LOCAL_BGE_M3}, "
                         f"否则用 HF id {_DEFAULT_HF_ID}")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default=None, help='e.g. "cuda:0"')
    ap.add_argument("--no-fp16", action="store_true",
                    help="禁用 fp16（GPU 下默认启用）")
    ap.add_argument("--strip-special", action="store_true",
                    help="剔除 [CLS]/[SEP]/<s>/</s> 等特殊 token")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None,
                    help="可选：将逐样本结果写到 JSON")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    preds = _load_texts(args.pred)
    refs = _load_texts(args.ref)
    if len(preds) != len(refs):
        raise SystemExit(
            f"[bge_score_f1_eval] 行数不一致: "
            f"preds={len(preds)} vs refs={len(refs)}"
        )

    ev = BGEScoreF1Evaluator(
        model_path=args.model_path,
        lang=args.lang,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=not args.no_fp16,
        strip_special=args.strip_special,
        verbose=args.verbose,
    )
    result = ev.score_detailed(preds, refs)
    P, R, F1 = result.mean

    print(
        f"[BGEScore-F1 | model={Path(ev.model_path).name} | "
        f"lang={ev.lang} | strip_special={ev.strip_special} | "
        f"N={len(preds)}]"
    )
    print(f"  Precision : {P:.4f}")
    print(f"  Recall    : {R:.4f}")
    print(f"  F1        : {F1:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric": "BGEScore-F1 (BERTScore formula + BGE-m3 backbone)",
            "model_path": ev.model_path,
            "lang": ev.lang,
            "strip_special": ev.strip_special,
            "max_length": ev.max_length,
            "n_samples": len(preds),
            **result.as_dict(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote per-sample scores to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
