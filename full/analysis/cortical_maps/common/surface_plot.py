import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
_GRAYBG = LinearSegmentedColormap.from_list('graybg', ['#9C9C9C', '#C7C7C7', '#E4E4E4'])
_HEAT = LinearSegmentedColormap.from_list('heat', plt.cm.YlOrBr(np.linspace(0.05, 0.8, 256)))

def _render(ax, coords, faces, values, vmin, curv, solid_color=None, side='lateral', vmax=None):
    av = np.abs(np.asarray(values, dtype=float))
    fx = coords[faces, 0].mean(1)
    med = np.median(coords[:, 0])
    if side == 'medial':
        faces_v = faces[fx >= med]
        xc = -coords[:, 1]
    else:
        faces_v = faces[fx < med]
        xc = coords[:, 1]
    tri = Triangulation(xc, coords[:, 2], faces_v)
    cmin, cmax = np.percentile(curv, [12, 88])
    if cmax - cmin < 1e-06:
        cmax = cmin + 1.0
    cn = np.clip((curv - cmin) / (cmax - cmin), 0.0, 1.0)
    cn = 1.0 / (1.0 + np.exp(-(cn - 0.5) * 6.0))
    vert_grey = 1.0 - cn
    ax.tripcolor(tri, vert_grey, cmap=_GRAYBG, vmin=0.0, vmax=1.0, shading='gouraud', edgecolors='none', rasterized=True)
    if vmax is None:
        eng = av[np.isfinite(av) & (av >= vmin)]
        ref = eng if eng.size > 20 else av[np.isfinite(av) & (av > 0)]
        vmax = float(np.percentile(ref, 75)) if ref.size else vmin + 1e-06
    vmax = max(vmax, vmin * 1.25 + 1e-09)
    fi = np.nan_to_num(av)[faces_v].mean(1)
    keep = fi >= vmin
    if keep.any():
        tri_s = Triangulation(xc, coords[:, 2], faces_v[keep])
        ax.tripcolor(tri_s, np.nan_to_num(av), cmap=_HEAT, norm=Normalize(vmin, vmax, clip=True), shading='gouraud', edgecolors='none', rasterized=True, alpha=0.88)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(xc.min() - 4, xc.max() + 4)
    ax.set_ylim(coords[:, 2].min() - 4, coords[:, 2].max() + 4)

def target_display_limits(values, visible_mask, target_coverage):
    visible = np.asarray(values, dtype=float)[visible_mask]
    finite = visible[np.isfinite(visible)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return (np.inf, np.inf, 0.0)
    footprint = 100.0 * positive.size / finite.size
    effective_target = min(float(target_coverage), footprint)
    threshold = float(np.percentile(finite, 100.0 - effective_target))
    if threshold <= 0:
        threshold = float(positive.min())
    engaged = finite[finite >= threshold]
    coverage = 100.0 * engaged.size / finite.size
    vmax = float(np.percentile(engaged, 75.0))
    vmax = max(vmax, threshold * 1.25 + 1e-12)
    return (threshold, vmax, coverage)

def engaged_vmax(values, visible_mask, threshold):
    visible = np.asarray(values, dtype=float)[visible_mask]
    engaged = visible[np.isfinite(visible) & (visible >= threshold)]
    if engaged.size == 0:
        return float(threshold * 1.25 + 1e-12)
    return max(float(np.percentile(engaged, 75.0)), float(threshold * 1.25 + 1e-12))
