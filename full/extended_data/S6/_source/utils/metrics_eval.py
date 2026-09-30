
























































from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["evaluate", "Record", "MetricEngine"]




DEFAULT_E5_PATH = "/home/guoyi/llm_model/multilingual-e5-large"
DEFAULT_BGE_PATH = "/home/guoyi/llm_model/bge-m3"


METRIC_COLUMNS = [
    "BGEScore", "BERTScore",
    "BLEU-1", "WER",
    "ROUGE-1", "ROUGE-2", "ROUGE-L",
    "METEOR",
]


_REF_KEYS = ["reference", "ref", "gt", "ground_truth", "groundtruth", "target",
             "label", "true", "truth", "gold", "original", "ref_text", "y_true"]
_HYP_KEYS = ["hypothesis", "hyp", "decoded", "prediction", "predicted", "pred",
             "output", "generated", "gen", "candidate", "cand", "hyp_text",
             "y_pred"]
_SUBJ_KEYS = ["subject_name", "subject", "subj", "sub", "subject_id", "sid",
              "participant", "subid"]
_TASK_KEYS = ["task", "dataset", "story", "run_name"]
_SPLIT_KEYS = ["split", "phase", "set"]

_SUBJ_PATH_RE = re.compile(r"(sub[-_][A-Za-z0-9]+)", re.IGNORECASE)
_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")





@dataclass
class Record:

    ref: str
    hyp: str
    subject: str = "unknown"
    task: str = ""
    split: str = ""
    source: str = ""
    lang: Optional[str] = None






_FR_STOP = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "que",
    "qui", "dans", "pour", "pas", "plus", "avec", "sur", "au", "aux", "ce",
    "cette", "ces", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "mais", "ne", "se", "sa", "son", "ses", "leur", "être", "avoir", "fait",
    "où", "ça", "été", "très", "comme", "moi", "toi", "lui",
}
_EN_STOP = {
    "the", "a", "an", "and", "is", "are", "was", "were", "to", "of", "in",
    "that", "it", "for", "on", "with", "as", "this", "but", "be", "have",
    "has", "i", "you", "he", "she", "we", "they", "not", "at", "by", "so",
}
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def detect_lang(text: str) -> str:

    if not text:
        return "en"
    cjk = len(_CJK_RE.findall(text))
    letters = sum(c.isalpha() for c in text)
    if letters == 0:
        return "zh" if cjk else "en"
    if cjk / max(letters, 1) > 0.15:
        return "zh"

    low = text.lower()
    toks = _WORD_RE.findall(low)
    if not toks:
        return "en"
    fr_hits = sum(t in _FR_STOP for t in toks)
    en_hits = sum(t in _EN_STOP for t in toks)
    accent = len(re.findall(r"[àâäçéèêëîïôöùûüœ]", low))
    fr_score = fr_hits + (accent > 0) * 2
    if fr_score > en_hits:
        return "fr"
    return "en"


def tokenize(text: str, lang: str) -> List[str]:

    if lang == "zh":

        return [c for c in re.sub(r"\s+", "", text) if c.strip()]
    return _WORD_RE.findall(text.lower())


def clean_for_embedding(text: str, lang: str) -> str:

    if lang == "zh":
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()





def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(ref_tok: Sequence[str], hyp_tok: Sequence[str]) -> float:
    if len(ref_tok) == 0:
        return 0.0 if len(hyp_tok) == 0 else 1.0
    return _levenshtein(ref_tok, hyp_tok) / len(ref_tok)


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i:i + n])] += 1
    return counts


def bleu1(ref_tok: Sequence[str], hyp_tok: Sequence[str]) -> float:

    h = len(hyp_tok)
    r = len(ref_tok)
    if h == 0:
        return 0.0
    ref_counts = _ngram_counts(ref_tok, 1)
    hyp_counts = _ngram_counts(hyp_tok, 1)
    overlap = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())

    prec = (overlap + 1.0) / (h + 1.0)
    bp = 1.0 if h > r else math.exp(1.0 - r / max(h, 1))
    return bp * prec


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def _prf(match: float, hyp_total: int, ref_total: int) -> float:
    if hyp_total == 0 or ref_total == 0 or match == 0:
        return 0.0
    p = match / hyp_total
    r = match / ref_total
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def rouge_n(ref_tok: Sequence[str], hyp_tok: Sequence[str], n: int) -> float:
    ref_counts = _ngram_counts(ref_tok, n)
    hyp_counts = _ngram_counts(hyp_tok, n)
    match = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
    return _prf(match, max(len(hyp_tok) - n + 1, 0), max(len(ref_tok) - n + 1, 0))


def rouge_l(ref_tok: Sequence[str], hyp_tok: Sequence[str]) -> float:
    lcs = _lcs_len(ref_tok, hyp_tok)
    return _prf(lcs, len(hyp_tok), len(ref_tok))



try:
    from nltk.stem.porter import PorterStemmer as _Porter
    _STEMMER = _Porter()
except Exception:
    _STEMMER = None


def _meteor_align(ref_tok: List[str], hyp_tok: List[str], lang: str):




    matches: List[Tuple[int, int]] = []
    ref_used = [False] * len(ref_tok)
    hyp_used = [False] * len(hyp_tok)

    def _pass(transform):
        ref_t = [transform(t) for t in ref_tok]
        hyp_t = [transform(t) for t in hyp_tok]
        ref_map: Dict[str, List[int]] = defaultdict(list)
        for j, t in enumerate(ref_t):
            ref_map[t].append(j)
        for i, t in enumerate(hyp_t):
            if hyp_used[i]:
                continue
            for j in ref_map.get(t, []):
                if not ref_used[j]:
                    hyp_used[i] = True
                    ref_used[j] = True
                    matches.append((i, j))
                    break

    _pass(lambda t: t)
    if lang == "en" and _STEMMER is not None:
        _pass(lambda t: _STEMMER.stem(t))

    matches.sort()
    return matches


def meteor(ref_tok: List[str], hyp_tok: List[str], lang: str,
           alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    if len(ref_tok) == 0 or len(hyp_tok) == 0:
        return 0.0
    matches = _meteor_align(ref_tok, hyp_tok, lang)
    m = len(matches)
    if m == 0:
        return 0.0
    p = m / len(hyp_tok)
    r = m / len(ref_tok)
    fmean = p * r / (alpha * p + (1 - alpha) * r) if (alpha * p + (1 - alpha) * r) > 0 else 0.0

    chunks = 0
    prev = None
    for hi, rj in matches:
        if prev is None or hi != prev[0] + 1 or rj != prev[1] + 1:
            chunks += 1
        prev = (hi, rj)
    penalty = gamma * (chunks / m) ** beta
    return fmean * (1 - penalty)





class _E5BertScore:


    def __init__(self, model_path: str, device: Optional[str], batch_size: int,
                 max_length: int = 512, use_fp16: bool = True, verbose: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        dtype = torch.float16 if (use_fp16 and "cuda" in str(self.device)) else torch.float32
        if verbose:
            print(f"[BERTScore] loading e5 backbone: {model_path} on {self.device}")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        try:
            self.model = AutoModel.from_pretrained(model_path, dtype=dtype)
        except TypeError:
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()

    def _encode_tokens(self, texts: List[str]):
        import torch.nn.functional as F
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state
            h = F.normalize(h.float(), p=2, dim=-1)
            mask = enc.attention_mask.bool()
            for b in range(h.size(0)):
                m = mask[b]
                vecs = h[b][m] if m.any() else h[b].mean(0, keepdim=True)
                out.append(vecs.cpu().numpy())
        return out

    def f1_pairs(self, hyps: List[str], refs: List[str]) -> List[float]:
        hv = self._encode_tokens(hyps)
        rv = self._encode_tokens(refs)
        scores = []
        for vc, vr in zip(hv, rv):
            sim = vc @ vr.T
            p = float(sim.max(axis=1).mean())
            r = float(sim.max(axis=0).mean())
            scores.append(2 * p * r / (p + r) if (p + r) > 1e-9 else 0.0)
        return scores


class _BgeDense:


    def __init__(self, model_path: str, device: Optional[str], batch_size: int,
                 max_length: int = 512, use_fp16: bool = True, verbose: bool = True):
        self.batch_size = batch_size
        self.max_length = max_length
        self._flag = None
        self._tf = None
        try:
            from FlagEmbedding import BGEM3FlagModel
            if verbose:
                print(f"[BGEScore] loading bge-m3 via FlagEmbedding: {model_path}")
            self._flag = BGEM3FlagModel(model_path, use_fp16=use_fp16,
                                        devices=device) if device else \
                BGEM3FlagModel(model_path, use_fp16=use_fp16)
        except Exception as e:
            if verbose:
                print(f"[BGEScore] FlagEmbedding 不可用（{type(e).__name__}），"
                      f"回退 transformers CLS pooling: {model_path}")
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if (use_fp16 and "cuda" in str(self.device)) else torch.float32
            self.tok = AutoTokenizer.from_pretrained(model_path)
            try:
                self.model = AutoModel.from_pretrained(model_path, dtype=dtype)
            except TypeError:
                self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
            self.model = self.model.to(self.device).eval()
            self._tf = True

    def _encode(self, texts: List[str]):
        import numpy as np
        if self._flag is not None:
            vecs = self._flag.encode(texts, batch_size=self.batch_size,
                                     max_length=self.max_length)["dense_vecs"]
            return np.asarray(vecs, dtype=np.float32)

        import torch.nn.functional as F
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state[:, 0]
            h = F.normalize(h.float(), p=2, dim=-1)
            out.append(h.cpu().numpy())
        return np.concatenate(out, axis=0)

    def cos_pairs(self, hyps: List[str], refs: List[str]) -> List[float]:
        hv = self._encode(hyps)
        rv = self._encode(refs)
        return [float((hv[i] * rv[i]).sum()) for i in range(len(hyps))]





def _pick(d: dict, keys: Sequence[str]) -> Optional[str]:
    lower = {k.lower().strip(): k for k in d.keys()}
    for cand in keys:
        if cand in lower:
            v = d[lower[cand]]
            if v is not None:
                return v
    return None


def _subject_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        mt = _SUBJ_PATH_RE.search(part)
        if mt:
            return mt.group(1)
    return path.stem


def _records_from_dict_rows(rows: List[dict], path: Path) -> List[Record]:
    recs: List[Record] = []
    path_subj = _subject_from_path(path)
    source = path.stem
    for row in rows:
        ref = _pick(row, _REF_KEYS)
        hyp = _pick(row, _HYP_KEYS)
        if ref is None or hyp is None:
            continue
        subj = _pick(row, _SUBJ_KEYS) or path_subj
        recs.append(Record(
            ref=str(ref), hyp=str(hyp), subject=str(subj),
            task=str(_pick(row, _TASK_KEYS) or ""),
            split=str(_pick(row, _SPLIT_KEYS) or ""),
            source=source,
        ))
    return recs


def parse_file(path: Path) -> List[Record]:

    ext = path.suffix.lower()
    try:
        if ext in (".jsonl", ".ndjson"):
            rows = []
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return _records_from_dict_rows(rows, path)
        if ext == ".json":
            with open(path, "r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for k in ("data", "results", "predictions", "items", "rows"):
                    if isinstance(obj.get(k), list):
                        obj = obj[k]
                        break
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return _records_from_dict_rows(obj, path)
            return []
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = _csv.DictReader(f, delimiter=delim)
                rows = [dict(r) for r in reader]
            return _records_from_dict_rows(rows, path)
    except Exception as e:
        print(f"[warn] 解析失败 {path}: {type(e).__name__}: {e}", file=sys.stderr)
    return []


def _gather_files(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.jsonl", "*.ndjson", "*.json", "*.csv", "*.tsv"):
                files.extend(sorted(p.rglob(ext)))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[warn] 路径不存在: {inp}", file=sys.stderr)
    uniq = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq





class MetricEngine:
    def __init__(self, e5_path: str = DEFAULT_E5_PATH, bge_path: str = DEFAULT_BGE_PATH,
                 device: Optional[str] = None, batch_size: int = 64,
                 compute_bert: bool = True, compute_bge: bool = True,
                 use_fp16: bool = True, verbose: bool = True):
        self.e5_path = e5_path
        self.bge_path = bge_path
        self.device = device
        self.batch_size = batch_size
        self.compute_bert = compute_bert
        self.compute_bge = compute_bge
        self.use_fp16 = use_fp16
        self.verbose = verbose


    @staticmethod
    def _lexical(ref: str, hyp: str, lang: str) -> Dict[str, float]:
        rt = tokenize(ref, lang)
        ht = tokenize(hyp, lang)
        return {
            "WER": wer(rt, ht),
            "BLEU-1": bleu1(rt, ht),
            "ROUGE-1": rouge_n(rt, ht, 1),
            "ROUGE-2": rouge_n(rt, ht, 2),
            "ROUGE-L": rouge_l(rt, ht),
            "METEOR": meteor(rt, ht, lang),
        }

    def run(self, records: List[Record], group_keys: Sequence[str] = ("subject",),
            lang_override: Optional[str] = None) -> "List[OrderedDict]":
        if not records:
            raise ValueError("没有可评测的句对（检查输入文件 / 字段名）。")


        for rec in records:
            rec.lang = lang_override or rec.lang or detect_lang(rec.ref or rec.hyp)


        per = []
        for rec in records:
            per.append(self._lexical(rec.ref, rec.hyp, rec.lang))


        if self.compute_bert:
            e5 = _E5BertScore(self.e5_path, self.device, self.batch_size,
                              use_fp16=self.use_fp16, verbose=self.verbose)
            hyps = [clean_for_embedding(r.hyp, r.lang) for r in records]
            refs = [clean_for_embedding(r.ref, r.lang) for r in records]
            for d, s in zip(per, e5.f1_pairs(hyps, refs)):
                d["BERTScore"] = s
            del e5
        if self.compute_bge:
            bge = _BgeDense(self.bge_path, self.device, self.batch_size,
                            use_fp16=self.use_fp16, verbose=self.verbose)
            hyps = [clean_for_embedding(r.hyp, r.lang) for r in records]
            refs = [clean_for_embedding(r.ref, r.lang) for r in records]
            for d, s in zip(per, bge.cos_pairs(hyps, refs)):
                d["BGEScore"] = s
            del bge


        groups: "OrderedDict[tuple, List[int]]" = OrderedDict()
        for i, rec in enumerate(records):
            key = tuple(getattr(rec, k) for k in group_keys)
            groups.setdefault(key, []).append(i)

        active_metrics = [m for m in METRIC_COLUMNS
                          if (m not in ("BERTScore",) or self.compute_bert)
                          and (m not in ("BGEScore",) or self.compute_bge)]

        rows: List[OrderedDict] = []
        for key, idxs in groups.items():
            row: "OrderedDict[str, object]" = OrderedDict()
            for k, v in zip(group_keys, key):
                row[k] = v
            row["lang"] = records[idxs[0]].lang
            row["n"] = len(idxs)
            for m in active_metrics:
                vals = [per[i][m] for i in idxs if m in per[i]]
                row[m] = float(sum(vals) / len(vals)) if vals else float("nan")
            rows.append(row)


        rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in group_keys))
        return rows





def _append_summary(df, group_keys: Sequence[str], metric_cols: Sequence[str]):





    import pandas as pd

    if len(df) == 0:
        return df
    numeric_cols = [c for c in (["n"] + list(metric_cols)) if c in df.columns]
    label_col = group_keys[0]
    other_cols = [c for c in df.columns if c not in numeric_cols]

    def _summary_row(label, series_func):
        row = {c: "" for c in other_cols}
        row[label_col] = label
        for c in numeric_cols:
            row[c] = series_func(df[c])
        return row

    mean_row = _summary_row("MEAN", lambda s: float(s.mean()))
    std_row = _summary_row("STD", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0)
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def evaluate(inputs: Sequence[str], out_csv: Optional[str] = None,
             lang: Optional[str] = None, group_by: Sequence[str] = ("subject",),
             e5_path: str = DEFAULT_E5_PATH, bge_path: str = DEFAULT_BGE_PATH,
             device: Optional[str] = None, batch_size: int = 64,
             compute_bert: bool = True, compute_bge: bool = True,
             split: Optional[str] = None, use_fp16: bool = True,
             summary: bool = True, verbose: bool = True):











    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    if lang and lang.lower() == "auto":
        lang = None

    files = _gather_files(inputs)
    records: List[Record] = []
    for f in files:
        recs = parse_file(f)
        if recs and verbose:
            print(f"[parse] {f}  -> {len(recs)} pairs")
        records.extend(recs)

    if split:
        records = [r for r in records if r.split == split]

    engine = MetricEngine(e5_path=e5_path, bge_path=bge_path, device=device,
                          batch_size=batch_size, compute_bert=compute_bert,
                          compute_bge=compute_bge, use_fp16=use_fp16, verbose=verbose)
    rows = engine.run(records, group_keys=tuple(group_by), lang_override=lang)

    import pandas as pd
    df = pd.DataFrame(rows)
    if summary:
        active = [m for m in METRIC_COLUMNS if m in df.columns]
        df = _append_summary(df, tuple(group_by), active)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[done] {len(df)} groups -> {out_csv}")
    return df





def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="按受试者级别计算 WER/BLEU-1/ROUGE/METEOR/BERTScore(e5)/"
                    "BGEScore(bge-m3)，输出一行一受试者的 CSV。")
    ap.add_argument("inputs", nargs="+", help="一个或多个文件/目录（jsonl/json/csv 可混合）")
    ap.add_argument("-o", "--out", required=True, help="输出 CSV 路径")
    ap.add_argument("--lang", default="auto",
                    help="auto(默认,自动判定) | zh | en | fr（强制全部为某语言）")
    ap.add_argument("--group-by", default="subject",
                    help="逗号分隔分组键，默认 subject；可选 subject,source,task,split")
    ap.add_argument("--split", default=None, help="仅评测某 split（如 test/val）")
    ap.add_argument("--device", default=None, help='如 "cuda:0"；缺省自动')
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--e5-path", default=DEFAULT_E5_PATH)
    ap.add_argument("--bge-path", default=DEFAULT_BGE_PATH)
    ap.add_argument("--no-bert", action="store_true", help="跳过 BERTScore(e5)")
    ap.add_argument("--no-bge", action="store_true", help="跳过 BGEScore(bge-m3)")
    ap.add_argument("--no-summary", action="store_true",
                    help="不追加表尾 MEAN / STD 两行")
    ap.add_argument("--no-fp16", action="store_true", help="禁用 fp16")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    df = evaluate(
        args.inputs, out_csv=args.out, lang=args.lang,
        group_by=[k.strip() for k in args.group_by.split(",") if k.strip()],
        e5_path=args.e5_path, bge_path=args.bge_path, device=args.device,
        batch_size=args.batch_size, compute_bert=not args.no_bert,
        compute_bge=not args.no_bge, split=args.split,
        summary=not args.no_summary, use_fp16=not args.no_fp16,
        verbose=not args.quiet,
    )
    if not args.quiet:
        import pandas as pd
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
