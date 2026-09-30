


















from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"
SPLITS_JSON = SCRIPT_DIR / "seedv_cross_splits.json"



PYTHON_BD = _sys.executable
PYTHON_LB = _sys.executable

EIGEN_CLASSIFY = str(_LOCAL_SOURCE / 'guoyi_exp/Exp1/classify_baseline_v2.py')
EIGEN_PRIOR_CKPT = str(_LOCAL_SOURCE / 'checkpoints/unified_v3/prior_unified_v3_20260514_020311.pt')
SEEDV_D_PATH = str(_LOCAL_SOURCE / 'guoyi_exp/data/eigenmode_test/SEED-V/sensor_geom_eigenmode_template_60x48.npy')
SEEDV_RATIO = "1e-3"

LABRAM_DRIVER  = SCRIPT_DIR / "labram_perturb_run.py"
CBRAMOD_DRIVER = SCRIPT_DIR / "cbramod_perturb_run.py"

TAGS = [
    "clean",
    "noise_03", "noise_05", "noise_10",
    "drop_01",  "drop_03",  "drop_05",
    "red_50",   "red_25",   "red_10",
]
N_FOLDS = 5
N_CLASSES = 5


def make_eigen_cmd(tag, gpu, out_root, run_id, args):
    h5 = DATA_ROOT / tag
    out_dir = out_root / "eigen"
    return [
        "bash", "-c",

        f"ulimit -n 65536; cd /media/wsqlab/data/gy/code/Eigen_brain_decoding/guoyi_exp/Exp1 && "
        f"{PYTHON_BD} {EIGEN_CLASSIFY} "
        f"--h5_dir {h5} "
        f"--encoder_ckpt {EIGEN_PRIOR_CKPT} "
        f"--d_path {SEEDV_D_PATH} --ratio {SEEDV_RATIO} "
        f"--cv_mode cross --train_mode frozen --cache_features "
        f"--label_key labels --n_classes {N_CLASSES} --modality eeg "
        f"--n_folds {N_FOLDS} --seed 0 "
        f"--epochs {args.eigen_epochs} --batch_size {args.eigen_batch_size} "
        f"--patience {args.eigen_patience} "
        f"--tag SEEDV_{tag} "
        f"--run_id {run_id}_{tag} "
        f"--out_dir {out_dir} "
        f"--device cuda:{gpu}"
    ]


def make_labram_cmd(tag, gpu, out_root, args):
    h5 = DATA_ROOT / tag
    out_dir = out_root / "labram" / f"SEEDV_{tag}_cross5fold"
    return [
        PYTHON_LB, str(LABRAM_DRIVER),
        "--h5_dir", str(h5), "--tag", tag,
        "--out_dir", str(out_dir),
        "--splits_json", str(SPLITS_JSON),
        "--cuda", str(gpu),
        "--seed", str(args.labram_seed),
        "--epochs", str(args.labram_epochs),
        "--batch_size", str(args.labram_batch_size),
        "--patience", str(args.labram_patience),
        "--frozen",
    ]


def make_cbramod_cmd(tag, gpu, out_root, args):
    h5 = DATA_ROOT / tag
    out_dir = out_root / "cbramod" / f"SEEDV_{tag}_cross5fold"
    return [
        PYTHON_BD, str(CBRAMOD_DRIVER),
        "--h5_dir", str(h5), "--tag", tag,
        "--out_dir", str(out_dir),
        "--splits_json", str(SPLITS_JSON),
        "--cuda", str(gpu),
        "--seed", str(args.cbramod_seed),
        "--epochs", str(args.cbramod_epochs),
        "--batch_size", str(args.cbramod_batch_size),
        "--patience", str(args.cbramod_patience),
        "--frozen",
    ]


def eigen_done(out_root, tag, run_id):

    run_dir_prefix = out_root / "eigen" / f"SEEDV_{tag}_eeg_cross_frozen_cache_{run_id}_{tag}"
    return (run_dir_prefix / "summary.json").is_file()


def labram_done(out_root, tag):
    return (out_root / "labram" / f"SEEDV_{tag}_cross5fold" / "summary.json").is_file()


def cbramod_done(out_root, tag):
    return (out_root / "cbramod" / f"SEEDV_{tag}_cross5fold" / "summary.json").is_file()


def run_job(method, tag, gpu, out_root, run_id, args, log_q):
    log_path = SCRIPT_DIR / "logs" / f"{method}_{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if method == "eigen":
        cmd = make_eigen_cmd(tag, gpu, out_root, run_id, args)
    elif method == "labram":
        cmd = make_labram_cmd(tag, gpu, out_root, args)
    elif method == "cbramod":
        cmd = make_cbramod_cmd(tag, gpu, out_root, args)
    else:
        raise ValueError(method)
    t0 = time.time()
    log_q.put(("INFO", f"[{method:<7} {tag:<9} cuda:{gpu}] START  → {log_path}"))
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n"); lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60.0
    if proc.returncode == 0:
        log_q.put(("OK",   f"[{method:<7} {tag:<9} cuda:{gpu}] DONE   in {dt:.1f} min"))
    else:
        log_q.put(("FAIL", f"[{method:<7} {tag:<9} cuda:{gpu}] FAILED rc={proc.returncode} ({dt:.1f} min) → see {log_path}"))


def worker_loop(wid, gpu, method, job_q, log_q, out_root, run_id, args, results):
    while True:
        try:
            tag = job_q.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(method, tag, gpu, out_root, run_id, args, log_q)
            results.append((method, tag, "ok"))
        except Exception as e:
            log_q.put(("FAIL", f"[w{wid} {method} {tag}] EXC {e!r}"))
            results.append((method, tag, f"exc:{e!r}"))
        finally:
            job_q.task_done()


def log_consumer(log_q, summary_path):
    with open(summary_path, "a", buffering=1) as f:
        while True:
            item = log_q.get()
            if item is None:
                return
            level, msg = item
            line = f"{datetime.now().strftime('%H:%M:%S')} [{level}] {msg}"
            print(line, flush=True); f.write(line + "\n")
            log_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["eigen", "labram", "cbramod"],
                    choices=["eigen", "labram", "cbramod"])
    ap.add_argument("--tags", nargs="+", default=None,
                    help="Subset of perturbation tags (default: all 10).")
    ap.add_argument("--plan", default="eigen:0,labram:1,cbramod:0",
                    help="method:gpu_id assignments. Each method gets one worker on the given GPU.")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the command list and exit.")


    ap.add_argument("--eigen_epochs", type=int, default=50)
    ap.add_argument("--eigen_batch_size", type=int, default=32)
    ap.add_argument("--eigen_patience", type=int, default=10)


    ap.add_argument("--labram_epochs", type=int, default=30)
    ap.add_argument("--labram_batch_size", type=int, default=64)
    ap.add_argument("--labram_patience", type=int, default=8)
    ap.add_argument("--labram_seed", type=int, default=3407)


    ap.add_argument("--cbramod_epochs", type=int, default=30)
    ap.add_argument("--cbramod_batch_size", type=int, default=64)
    ap.add_argument("--cbramod_patience", type=int, default=8)
    ap.add_argument("--cbramod_seed", type=int, default=3407)

    args = ap.parse_args()

    out_root = SCRIPT_DIR / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / "logs").mkdir(exist_ok=True)
    if args.run_id is None:
        args.run_id = "tyf_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    plan = {}
    for tok in args.plan.split(","):
        method, gpu = tok.split(":")
        if method not in args.methods:
            continue
        plan[method] = int(gpu)
    for m in args.methods:
        plan.setdefault(m, 0)

    tags = args.tags or TAGS
    summary_path = SCRIPT_DIR / "logs" / "orchestrator.log"


    per_method_jobs = {m: [] for m in args.methods}
    skipped = []
    for m in args.methods:
        for tag in tags:
            done_fn = {"eigen": lambda t=tag: eigen_done(out_root, t, args.run_id),
                       "labram": lambda t=tag: labram_done(out_root, t),
                       "cbramod": lambda t=tag: cbramod_done(out_root, t)}[m]
            if done_fn():
                skipped.append((m, tag)); continue
            per_method_jobs[m].append(tag)

    n_total = sum(len(v) for v in per_method_jobs.values())
    print(f"[orch] run_id   = {args.run_id}")
    print(f"[orch] methods  = {args.methods}")
    print(f"[orch] tags     = {tags}")
    print(f"[orch] plan     = {plan}  (one worker per method on the given GPU)")
    print(f"[orch] skipped  = {len(skipped)} (summary.json already present)")
    for m, t in skipped:
        print(f"          - {m} {t}")
    print(f"[orch] todo     = {n_total} jobs total")
    for m, ts in per_method_jobs.items():
        print(f"          {m}: {ts}")

    if args.dry_run:
        print("\n[dry_run] commands that would be issued:")
        for m in args.methods:
            gpu = plan[m]
            for tag in per_method_jobs[m]:
                if m == "eigen":
                    cmd = make_eigen_cmd(tag, gpu, out_root, args.run_id, args)
                elif m == "labram":
                    cmd = make_labram_cmd(tag, gpu, out_root, args)
                else:
                    cmd = make_cbramod_cmd(tag, gpu, out_root, args)
                print("\n  $ " + (" ".join(cmd) if isinstance(cmd[0], str) and cmd[0] != "bash"
                                  else cmd[2]))
        return

    if n_total == 0:
        print("[orch] nothing to do.")
        return

    job_qs = {m: queue.Queue() for m in args.methods}
    for m, ts in per_method_jobs.items():
        for t in ts:
            job_qs[m].put(t)
    log_q = queue.Queue()
    results = []
    consumer = threading.Thread(target=log_consumer, args=(log_q, summary_path), daemon=True)
    consumer.start()

    threads = []
    for wid, m in enumerate(args.methods):
        if job_qs[m].empty():
            continue
        gpu = plan[m]
        t = threading.Thread(target=worker_loop,
                             args=(wid, gpu, m, job_qs[m], log_q, out_root, args.run_id, args, results),
                             daemon=True)
        t.start(); threads.append(t)

    for t in threads:
        t.join()
    log_q.put(None); consumer.join(timeout=2.0)

    n_ok = sum(1 for r in results if r[2] == "ok")
    n_fail = len(results) - n_ok
    print(f"\n[orch] DONE: {n_ok} ok, {n_fail} failed (of {len(results)} jobs run).")
    if n_fail:
        for m, t, st in results:
            if st != "ok":
                print(f"  FAILED  {m} {t}  → {st}")


if __name__ == "__main__":
    main()
