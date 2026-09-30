
import argparse, os, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); EXP=HERE.parents[1]; BASE=HERE.parents[2]
CLASSIFIER=BASE.parent/"Exp_Classification"/"code"/"classify_baseline_v2.py"
CKPT=EXP/"pretrain"/"prior_fourier_seedv_mae50.pt"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--paradigm",choices=["within","pool"],required=True)
    ap.add_argument("--seed",type=int,choices=range(10),required=True); ap.add_argument("--device",required=True)
    ap.add_argument("--experiment-dir",default=str(EXP)); ap.add_argument("--mode-name",default="fourier"); a=ap.parse_args()
    exp=Path(a.experiment_dir).resolve(); ckpt=exp/"pretrain"/f"prior_{a.mode_name}_seedv_mae50.pt"
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    out=exp/"result"/a.paradigm/f"seed_{a.seed:02d}"
    if list(out.glob("*/summary.json")): print(f"SKIP completed {a.paradigm} seed {a.seed}"); return
    out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(CLASSIFIER),"--h5_dir",str(exp/"data"),"--encoder_ckpt",str(ckpt),
      "--train_mode","frozen","--cache_features","--cv_mode","within" if a.paradigm=="within" else "cross",
      "--modality","eeg","--label_key","labels","--n_folds","5","--epochs","200","--batch_size","32",
      "--lr_adapter","0.001","--lr_head","0.001","--lat_dim","256","--dropout","0.3","--patience","30",
      "--kl_weight","0.001","--n_classes","5","--device",a.device,"--seed",str(a.seed),
      "--tag",f"SEEDV_{a.mode_name}_matched_{a.paradigm}_seed{a.seed:02d}","--out_dir",str(out),"--run_id","run"]
    env=os.environ.copy(); env.setdefault("OMP_NUM_THREADS","4"); subprocess.run(cmd,check=True,env=env)

if __name__=="__main__":
    import sys
    main()
