































from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import bert_score


__all__ = [
    "calculate_bert_score",
    "BERTScoreEvaluator",
    "ScoreResult",
]


_SUPPORTED_LANGS = {"en", "zh"}


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
        raise ValueError("lang must be provided ('en' or 'zh').")
    l = lang.strip().lower()
    if l in {"en", "english", "eng"}:
        return "en"
    if l in {"zh", "zh_cn", "zh-cn", "chinese", "cn"}:
        return "zh"
    raise ValueError(f"Unsupported lang={lang!r}. Supported: {_SUPPORTED_LANGS}.")


def _coerce_text_list(x) -> List[str]:
    if isinstance(x, (tuple, list)):
        return [str(s) for s in x]
    raise TypeError(f"Expected list/tuple of strings, got {type(x).__name__}.")


class BERTScoreEvaluator:


    def __init__(
        self,
        lang: str = "en",
        model_type: Optional[str] = None,
        num_layers: Optional[int] = None,
        rescale_with_baseline: Optional[bool] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
        nthreads: int = 4,
        verbose: bool = False,
        idf: bool = False,
    ):
        self.lang = _normalize_lang(lang)


        if rescale_with_baseline is None:
            rescale_with_baseline = (self.lang == "en")
        self.rescale_with_baseline = rescale_with_baseline

        self.model_type = model_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.nthreads = nthreads
        self.verbose = verbose
        self.idf = idf

        self._scorer = bert_score.BERTScorer(
            lang=self.lang,
            model_type=model_type,
            num_layers=num_layers,
            rescale_with_baseline=rescale_with_baseline,
            batch_size=batch_size,
            device=device,
            nthreads=nthreads,
            idf=idf,
        )

    def score(
        self,
        cands: Sequence[str],
        refs: Sequence[str],
    ) -> Tuple[float, float, float]:

        result = self.score_detailed(cands, refs)
        return result.mean

    def score_detailed(
        self,
        cands: Sequence[str],
        refs: Sequence[str],
    ) -> ScoreResult:
        cands = _coerce_text_list(cands)
        refs = _coerce_text_list(refs)
        if len(cands) != len(refs):
            raise ValueError(
                f"len(cands)={len(cands)} != len(refs)={len(refs)}."
            )
        if len(cands) == 0:
            raise ValueError("Empty input: no candidate/reference pairs.")
        P, R, F1 = self._scorer.score(cands, refs, verbose=self.verbose)
        return ScoreResult(P, R, F1)


def calculate_bert_score(
    reconstructed_texts: Union[Sequence[str], Iterable[str]],
    original_texts: Union[Sequence[str], Iterable[str]],
    lang: str = "en",
    rescale_with_baseline: Optional[bool] = None,
    model_type: Optional[str] = None,
    num_layers: Optional[int] = None,
    batch_size: int = 64,
    device: Optional[str] = None,
    verbose: bool = False,
    idf: bool = False,
) -> Tuple[float, float, float]:




















    reconstructed_texts = _coerce_text_list(list(reconstructed_texts))
    original_texts = _coerce_text_list(list(original_texts))

    lang_n = _normalize_lang(lang)
    if rescale_with_baseline is None:
        rescale_with_baseline = (lang_n == "en")

    P, R, F1 = bert_score.score(
        reconstructed_texts,
        original_texts,
        lang=lang_n,
        model_type=model_type,
        num_layers=num_layers,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
        idf=idf,
    )
    return (
        float(P.mean().item()),
        float(R.mean().item()),
        float(F1.mean().item()),
    )




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
        description="Compute BERTScore between predicted and reference texts "
                    "(supports English and Chinese)."
    )
    ap.add_argument("--pred", required=True, help="预测文本文件 (.txt / .json)")
    ap.add_argument("--ref", required=True, help="参考文本文件 (.txt / .json)")
    ap.add_argument("--lang", default="en", choices=["en", "zh", "EN", "ZH", "zh_CN"],
                    help="语言: en (英文) 或 zh (中文)")
    ap.add_argument("--model-type", default=None,
                    help="可选：显式指定 HF 模型名 (覆盖 lang 默认)")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help='e.g. "cuda:0"')
    ap.add_argument("--no-rescale", action="store_true",
                    help="禁用 rescale_with_baseline（中文默认已禁用）")
    ap.add_argument("--rescale", action="store_true",
                    help="强制启用 rescale_with_baseline")
    ap.add_argument("--idf", action="store_true", help="启用 IDF 加权")
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
            f"[bert_score_eval] 行数不一致: preds={len(preds)} vs refs={len(refs)}"
        )

    if args.rescale and args.no_rescale:
        raise SystemExit("--rescale 与 --no-rescale 互斥")
    rescale: Optional[bool] = None
    if args.rescale:
        rescale = True
    elif args.no_rescale:
        rescale = False

    ev = BERTScoreEvaluator(
        lang=args.lang,
        model_type=args.model_type,
        num_layers=args.num_layers,
        rescale_with_baseline=rescale,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose,
        idf=args.idf,
    )
    result = ev.score_detailed(preds, refs)
    P, R, F1 = result.mean

    print(f"[BERTScore | lang={ev.lang} | rescale={ev.rescale_with_baseline} | N={len(preds)}]")
    print(f"  Precision : {P:.4f}")
    print(f"  Recall    : {R:.4f}")
    print(f"  F1        : {F1:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lang": ev.lang,
            "rescale_with_baseline": ev.rescale_with_baseline,
            "n_samples": len(preds),
            **result.as_dict(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote per-sample scores to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
