








from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, h5py
from sklearn.model_selection import StratifiedShuffleSplit

SPARC_DIR = "/home/guoyi/a800/share/code/Eigen_brain_decoding/data_check_20260507/uncertainty_seedv/tyf_20260518"
sys.path.insert(0, SPARC_DIR)
import sparcnet_run as S
HERE = Path(__file__).resolve().parent
RUN = HERE / "data_ecog_eigen_bp/run_20260529_215039"
SIM = RUN / "feat_similarity"
N_CLS = 6


def day_centroid_similarity(F, y, sess):
    days = sorted(np.unique(sess).tolist())
    cent = np.full((len(days), N_CLS, F.shape[1]), np.nan)
    for di, d in enumerate(days):
        for c in range(N_CLS):
            m = (sess == d) & (y == c)
            if m.sum() > 0: cent[di, c] = F[m].mean(0)
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    Sim = np.zeros((len(days), len(days)))
    for i in range(len(days)):
        for j in range(len(days)):
            cs = [cos(cent[i, c], cent[j, c]) for c in range(N_CLS)
                  if not (np.isnan(cent[i, c]).any() or np.isnan(cent[j, c]).any())]
            Sim[i, j] = np.mean(cs)
    return Sim, days


@torch.no_grad()
def penult(model, X, dev, bs=128):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).to(dev)
        out.append(model.densenet(xb).flatten(1).float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--src_sr", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    split = json.load(open(RUN / "split.json"))["split"]
    with h5py.File(RUN / "sub-01.h5", "r") as f:
        eeg = f["eeg_raw"][:].astype(np.float32); y = f["labels"][:].astype(np.int64); sess = f["session"][:]
    X = S.zscore(S.resample_to(S.TARGET_SR, eeg, a.src_sr))

    import logging; logging.basicConfig(level=logging.INFO); lg = logging.getLogger()
    tr_all = np.array(split["train"])
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=a.seed)
    cr, vr = next(sss.split(tr_all, y[tr_all])); tr, va = tr_all[cr], tr_all[vr]
    args = type("A", (), dict(epochs=a.epochs, batch_size=64, lr=5e-4, weight_decay=1e-4,
               label_smoothing=0.1, clip_value=1.0, patience=12, drop_rate=0.2, drop_fc=0.5,
               growth_rate=32, block_config=[4,4,4,4,4,4,4], num_init_features=64, n_classes=6))()
    _, state = S.train_one_fold(X, y, tr, va, va, 6, dev, args, lg)
    model = S.SPaRCNet(in_channels=X.shape[1], num_classes=6, growth_rate=32,
                       block_config=(4,4,4,4,4,4,4), num_init_features=64,
                       drop_rate=0.2, drop_fc=0.5, batch_norm=True, conv_bias=False).to(dev)
    model.load_state_dict(state)

    obs_feat = penult(model, X, dev)
    np.save(SIM / "obs_feat.npy", obs_feat.astype(np.float32))
    Sob, days = day_centroid_similarity(obs_feat, y, sess)
    np.save(SIM / "obs_similarity.npy", Sob)
    print("obs sim diag", round(np.diag(Sob).mean(),3), "off", round(Sob[~np.eye(len(days),dtype=bool)].mean(),3),
          "d0-d211", round(Sob[0,-1],3))


    pri = np.load(SIM / "feat_z.npy")
    zob = (obs_feat - obs_feat.mean(0)) / (obs_feat.std(0) + 1e-6)
    zpr = (pri - pri.mean(0)) / (pri.std(0) + 1e-6)
    fused = np.concatenate([zob, zpr], 1)
    Sfu, _ = day_centroid_similarity(fused, y, sess)
    np.save(SIM / "fused_similarity.npy", Sfu)
    print("fused sim diag", round(np.diag(Sfu).mean(),3), "off", round(Sfu[~np.eye(len(days),dtype=bool)].mean(),3),
          "d0-d211", round(Sfu[0,-1],3))
    print("[saved] obs_similarity.npy / fused_similarity.npy")


if __name__ == "__main__":
    main()
