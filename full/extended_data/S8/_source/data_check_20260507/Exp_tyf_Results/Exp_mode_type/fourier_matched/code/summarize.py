
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve(); EXP=HERE.parents[1]; RES=EXP/"result"; subjects=[f"sub-{i:02d}" for i in range(1,17)]

def crossfit_pred(z1,z2,y):
    grid=np.linspace(.05,10,200); A=[];B=[]; rng=np.random.default_rng(0)
    for c in range(5):
        ix=np.where(y==c)[0]; rng.shuffle(ix); m=len(ix)//2; A.extend(ix[:m]); B.extend(ix[m:])
    A=np.array(sorted(A));B=np.array(sorted(B)); pred=np.empty(len(y),int)
    def temp(z,yy):
        z=torch.from_numpy(z).float(); yy=torch.from_numpy(yy).long(); best=(1.,float("inf"))
        for t in grid:
            loss=float(-F.log_softmax(z/t,-1)[torch.arange(len(yy)),yy].mean())
            if loss<best[1]: best=(float(t),loss)
        return best[0]
    for fit,use in ((A,B),(B,A)):
        t1=temp(z1[fit],y[fit]);t2=temp(z2[fit],y[fit])
        pred[use]=(F.log_softmax(torch.from_numpy(z1[use]).float()/t1,-1)+F.log_softmax(torch.from_numpy(z2[use]).float()/t2,-1)).argmax(-1).numpy()
    return pred

def find_summary(p):
    x=list(p.glob("*/summary.json")); return x[0] if len(x)==1 else None

def main():
    global EXP,RES
    ap=argparse.ArgumentParser(); ap.add_argument("--experiment-dir",default=str(EXP)); ap.add_argument("--title",default="傅里叶基"); a=ap.parse_args()
    EXP=Path(a.experiment_dir).resolve(); RES=EXP/"result"
    rows=[]; matrices={"within":{},"pool":{}}
    for seed in range(10):
        sp=find_summary(RES/"within"/f"seed_{seed:02d}")
        if sp:
            d=json.loads(sp.read_text())
            for sub in subjects:
                acc=d["per_subject"][sub]["poe_temp"]["mean"]
                rows.append(["within",seed,sub,acc,5]); matrices["within"].setdefault(seed,{})[sub]=acc
        sp=find_summary(RES/"pool"/f"seed_{seed:02d}")
        if sp:
            root=sp.parent
            for fd in sorted(root.glob("fold_*")):
                npz=np.load(fd/"holdout_preds.npz"); sid=np.load(fd/"holdout_subj_ids.npy").astype(str)
                pred=crossfit_pred(npz["z_obs"],npz["z_pri"],npz["y"])
                for sub in sorted(set(sid)):
                    m=sid==sub; acc=float(np.mean(pred[m]==npz["y"][m])); n=int(m.sum())
                    rows.append(["pool",seed,sub,acc,n]); matrices["pool"].setdefault(seed,{})[sub]=acc
    RES.mkdir(parents=True,exist_ok=True)
    with open(RES/"seed_subject_accuracy.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["paradigm","seed","subject","mean_accuracy","n_folds_or_trials"]);w.writerows(rows)
    lines=[f"# SEED-V {a.title}本征模实验汇总","","主指标：温度校准 PoE 准确率。Within 为每位受试者 5 折准确率均值；Pool 为受试者留出折中该受试者全部测试样本的准确率。","",
      f"- 本征模类型：{a.title}，保留 30 模态","- Prior 位置编码：对应基的原生谱值加第一非零谱间隙（规避 log(0)）","- BT-ND：统一预训练权重初始化后，在 SEED-V 对应模态上进行 50 epoch MAE 适配","- 随机种子：0–9",""]
    for p,title in (("within","受试者内部 5 折"),("pool","受试者 Pool（跨受试者 5 折）")):
        lines += [f"## {title}",""]
        if matrices[p]:
            lines.append("|seed|"+"|".join(subjects)+"|seed均值|"); lines.append("|---|"+"|".join(["---:"]*17))
            for seed in sorted(matrices[p]):
                vals=[matrices[p][seed].get(s,float("nan")) for s in subjects]
                lines.append(f"|{seed}|"+"|".join("—" if np.isnan(v) else f"{100*v:.2f}%" for v in vals)+f"|{100*np.nanmean(vals):.2f}%|")
            allv=[v for x in matrices[p].values() for v in x.values()]
            lines += ["",f"当前已完成 {len(matrices[p])}/10 个 seed；seed×subject 记录 {len(allv)} 条；总体平均 {100*np.mean(allv):.2f}%。",""]
        else: lines += ["尚无已完成结果。",""]
    (RES/f"{a.title}本征模实验结果汇总.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"wrote {len(rows)} seed-subject rows")
if __name__=="__main__": main()
