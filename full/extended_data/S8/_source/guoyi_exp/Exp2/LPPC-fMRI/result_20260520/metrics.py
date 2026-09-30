










from __future__ import annotations
from typing import Sequence

import numpy as np


def _norm_lang(lang_code: str) -> str:
    return (lang_code or "EN").strip().upper()


def tokenize(text: str, lang_code: str) -> list[str]:
    text = text.strip()
    if _norm_lang(lang_code) in ("CN", "ZH"):
        return [c for c in text if not c.isspace()]
    return text.split()


def prep_for_space(text: str, lang_code: str) -> str:
    return " ".join(tokenize(text, lang_code))



def compute_wer(refs: Sequence[str], hyps: Sequence[str], lang_code: str) -> float:
    import jiwer
    refs_p = [prep_for_space(r, lang_code) or "_" for r in refs]
    hyps_p = [prep_for_space(h, lang_code) or "_" for h in hyps]
    try:
        return float(jiwer.wer(refs_p, hyps_p))
    except Exception:
        return float("nan")


def compute_bleu1(refs, hyps, lang_code: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    scores = []
    for r, h in zip(refs, hyps):
        rt = tokenize(r, lang_code); ht = tokenize(h, lang_code)
        if not ht:
            scores.append(0.0); continue
        try:
            s = sentence_bleu([rt], ht, weights=(1, 0, 0, 0),
                              smoothing_function=smooth)
        except Exception:
            s = 0.0
        scores.append(s)
    return float(np.mean(scores)) if scores else float("nan")


class _WhitespaceTokenizer:







    def tokenize(self, text):
        return text.split()


def compute_rouge(refs, hyps, lang_code: str) -> dict:
    from rouge_score import rouge_scorer
    if _norm_lang(lang_code) in ("CN", "ZH"):
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=False,
                                      tokenizer=_WhitespaceTokenizer())
    else:
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for r, h in zip(refs, hyps):
        rp = prep_for_space(r, lang_code) or "_"
        hp = prep_for_space(h, lang_code) or "_"
        try:
            s = sc.score(rp, hp)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        except Exception:
            r1.append(0.0); r2.append(0.0); rl.append(0.0)
    return {"ROUGE-1": float(np.mean(r1)) if r1 else float("nan"),
            "ROUGE-2": float(np.mean(r2)) if r2 else float("nan"),
            "ROUGE-L": float(np.mean(rl)) if rl else float("nan")}


def compute_meteor(refs, hyps, lang_code: str) -> float:
    lang = _norm_lang(lang_code)
    try:
        if lang in ("CN", "ZH"):


            import jieba
            from nltk.translate.meteor_score import single_meteor_score
            jieba.setLogLevel(20)
            scores = []
            for r, h in zip(refs, hyps):
                rt = [w.strip() for w in jieba.cut(str(r).strip()) if w.strip()]
                ht = [w.strip() for w in jieba.cut(str(h).strip()) if w.strip()]
                if not rt or not ht:
                    scores.append(0.0); continue
                try:
                    scores.append(single_meteor_score(rt, ht))
                except Exception:
                    scores.append(0.0)
            return float(np.mean(scores)) if scores else float("nan")

        from nltk.translate.meteor_score import meteor_score
        scores = []
        for r, h in zip(refs, hyps):
            rt = tokenize(r, "EN"); ht = tokenize(h, "EN")
            if not ht:
                scores.append(0.0); continue
            scores.append(meteor_score([rt], ht))
        return float(np.mean(scores)) if scores else float("nan")
    except Exception:
        return float("nan")


def compute_bertscore(refs, hyps, model_path: str, device: str,
                      lang_code: str, num_layers: int = 24) -> float:
    try:
        from bert_score import score as bscore
        P, R, F = bscore(hyps, refs, model_type=model_path,
                         num_layers=num_layers, device=device, verbose=False)
        return float(F.mean().item())
    except Exception:
        return float("nan")


def compute_bgescore(refs, hyps, bge_path: str, device: str) -> float:
    from sentence_transformers import SentenceTransformer
    import torch
    try:
        bge = SentenceTransformer(bge_path, device=device)
        pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
        if not pairs:
            return float("nan")
        rs, hs = zip(*pairs)
        re = bge.encode(list(rs), batch_size=32, convert_to_tensor=True,
                        normalize_embeddings=True)
        he = bge.encode(list(hs), batch_size=32, convert_to_tensor=True,
                        normalize_embeddings=True)
        sims = (re * he).sum(-1).cpu().numpy()
        del bge
        torch.cuda.empty_cache()
        return float(sims.mean())
    except Exception:
        return float("nan")



METRIC_KEYS = ("WER", "BLEU-1", "ROUGE-1", "ROUGE-2", "ROUGE-L",
               "METEOR", "BERTScore", "BGEScore")


def compute_all(refs: Sequence[str], hyps: Sequence[str],
                lang_code: str, bge_path: str,
                bertscore_model: str, device: str,
                bertscore_num_layers: int = 24) -> dict:
    n = sum(1 for r, h in zip(refs, hyps) if r and h)
    out = {"n": int(n)}
    out["WER"] = compute_wer(refs, hyps, lang_code)
    out["BLEU-1"] = compute_bleu1(refs, hyps, lang_code)
    out.update(compute_rouge(refs, hyps, lang_code))
    out["METEOR"] = compute_meteor(refs, hyps, lang_code)
    out["BERTScore"] = compute_bertscore(refs, hyps, bertscore_model, device,
                                          lang_code, bertscore_num_layers)
    out["BGEScore"] = compute_bgescore(refs, hyps, bge_path, device)
    return out
