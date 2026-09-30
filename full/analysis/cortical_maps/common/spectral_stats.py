import numpy as np
from scipy.stats import spearmanr, wilcoxon
NB = 12
x = np.arange(12)

def hier(D):
    rhos = np.array([spearmanr(x, D[s]).correlation for s in range(len(D))])
    rng = np.random.RandomState(0)
    bs = [np.nanmean(rhos[rng.randint(0, len(rhos), len(rhos))]) for _ in range(4000)]
    ci = np.nanpercentile(bs, [2.5, 97.5])
    p = wilcoxon(rhos).pvalue
    band = np.array([np.nanpercentile([np.nanmean(D[rng.randint(0, len(D), len(D)), b]) for _ in range(1500)], [2.5, 97.5]) for b in range(NB)])
    return (np.nanmean(rhos), ci, p, D.mean(0), band)
