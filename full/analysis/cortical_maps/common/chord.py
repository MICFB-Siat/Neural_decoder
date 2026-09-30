import os
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Wedge
from scipy.stats import wilcoxon
from common.runtime import TEMPLATES
from common import anatomy as bdata
NET7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
YEO = {'Vis': '#781286', 'SomMot': '#4682B4', 'DorsAttn': '#00760E', 'SalVentAttn': '#C43AFA', 'Limbic': '#C9A93B', 'Cont': '#E69422', 'Default': '#CD3E4E'}
YLAB = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default']
C_INK, C_MUT = ('#272727', '#767676')
ck = bdata.cortex_mask('lh')
co = bdata.surface('lh')[0]
PID = np.asarray(nib.load(str(TEMPLATES / 'parcels.dscalar.nii')).get_fdata()).ravel().astype(int)[:32492]
names = {k: v[0] for k, v in nib.load(str(TEMPLATES / 'networks.dlabel.nii')).header.get_axis(0).label[0].items()}

def net_of(p):
    nm = names.get(p, '')
    for i, n in enumerate(NET7):
        if n in nm:
            return i
    return -1
parcels = [p for p in range(1, 401) if (PID == p).any()]
nets = np.array([net_of(p) for p in parcels])
P = len(parcels)
cent = np.array([co[PID == p].mean(0) for p in parcels])
D = np.linalg.norm(cent[:, None] - cent[None], axis=2)
z = np.load(CACHE, allow_pickle=True)

def smooth(m):
    return np.where(ck, bdata.smooth_surface('lh', np.nan_to_num(m), iters=22), np.nan)

def imp(m):
    s = smooth(m)
    v = np.nan_to_num(np.array([np.nanmean(np.abs(s[PID == p])) for p in parcels]))
    return v / (v.max() + 1e-12)
IO = np.array([imp(z['maps_o'][i]) for i in range(12)])
IP = np.array([imp(z['maps_p'][i]) for i in range(12)])
COARSE_N = int(os.environ.get('CHORD_N', '105'))
from sklearn.cluster import KMeans
lab = np.full(P, -1, int)
nxt = 0
for ni in range(7):
    mem = np.where(nets == ni)[0]
    kc = max(1, int(round(len(mem) * COARSE_N / P)))
    if kc >= len(mem):
        for m in mem:
            lab[m] = nxt
            nxt += 1
        continue
    km = KMeans(n_clusters=kc, n_init=4, random_state=0).fit(cent[mem])
    for j, m in enumerate(mem):
        lab[m] = nxt + km.labels_[j]
    nxt += kc
S = nxt
agg = np.zeros((S, P))
for s in range(S):
    msk = lab == s
    agg[s, msk] = 1.0 / msk.sum()
nets = np.array([nets[lab == s][0] for s in range(S)])
cent = agg @ cent
D = np.linalg.norm(cent[:, None] - cent[None], axis=2)
IO = (agg @ IO.T).T
IP = (agg @ IP.T).T
P = S
print(f'coarsened to {P} supernodes  (per-network: {[int((nets == i).sum()) for i in range(7)]})')
good_p = np.array([IP[i].max() > 0.5 for i in range(12)])
io = IO.mean(0)
ip = IP[good_p].mean(0)
io /= io.max()
ip /= ip.max()
iu = np.triu_indices(P, 1)
dist = D[iu]
wl = lambda v: (np.outer(v, v)[iu] * dist).sum() / np.outer(v, v)[iu].sum()
WLO = np.array([wl(IO[i]) for i in range(12)])
WLP = np.array([wl(IP[i]) for i in good_p.nonzero()[0]])
pval = wilcoxon(WLP, WLO[good_p])[1]
print(f'weighted edge length: obs {WLO.mean():.1f}mm  prior {WLP.mean():.1f}mm  prior>obs {(WLP > WLO[good_p]).sum()}/{good_p.sum()}  p={pval:.2e}')
order, arc_of = ([], {})
gap = np.deg2rad(4)
ang = {}
start = np.pi / 2
total_gap = gap * 7
span = 2 * np.pi - total_gap
a = start
arc_bounds = []
for ni in range(7):
    members = [k for k in range(P) if nets[k] == ni]
    members.sort(key=lambda k: cent[k, 1])
    w = span * len(members) / P
    a0 = a
    for j, k in enumerate(members):
        ang[k] = a - (j + 0.5) * (w / max(len(members), 1))
    arc_bounds.append((ni, a0, a - w))
    a -= w + gap
R = 1.0
nx = np.array([R * np.cos(ang[k]) for k in range(P)])
ny = np.array([R * np.sin(ang[k]) for k in range(P)])
NODE_S = float(os.environ.get('CHORD_NODE_S', '16'))

def chord(ax, v, K, cmap_short, cmap_long):
    w = np.outer(v, v)
    w[np.tril_indices(P)] = 0
    flat = w.ravel()
    idx = np.argsort(flat)[::-1][:K]
    ii, jj = np.unravel_index(idx, w.shape)
    ww = flat[idx]
    wn = (ww - ww.min()) / (np.ptp(ww) + 1e-12)
    lens = D[ii, jj]
    o = np.argsort(lens)
    for t in o:
        i, j = (ii[t], jj[t])
        x0, y0, x1, y1 = (nx[i], ny[i], nx[j], ny[j])
        ln = lens[t]
        col = plt.cm.RdYlBu(np.clip((ln - 20) / 80, 0, 1))
        bow = 1.0 - 0.82 * np.clip(ln / 95, 0, 1)
        cx, cy = ((x0 + x1) / 2 * bow, (y0 + y1) / 2 * bow)
        pp = PathPatch(Path([(x0, y0), (cx, cy), (x1, y1)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc='none', ec=col, lw=0.35 + 0.9 * wn[t], alpha=0.55, zorder=2)
        ax.add_patch(pp)
    for ni, a0, a1 in arc_bounds:
        ax.add_patch(Wedge((0, 0), 1.13, np.rad2deg(a1), np.rad2deg(a0), width=0.07, fc=YEO[NET7[ni]], ec='white', lw=0.4, zorder=4))
        am = (a0 + a1) / 2
        ax.text(1.27 * np.cos(am), 1.27 * np.sin(am), YLAB[ni], rotation=0, ha='center', va='center', fontsize=5.6, color=YEO[NET7[ni]], fontweight='bold')
    ax.scatter(nx, ny, s=NODE_S, c=[YEO[NET7[n]] for n in nets], edgecolor='white', lw=0.3, zorder=5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.62, 1.45)
