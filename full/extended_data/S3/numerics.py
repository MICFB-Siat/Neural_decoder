
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

KEYS = ['within_rms_normalized','between_centroid_normalized','categorical_rsa_spearman']
DOMAINS = ['Motor','Perception','Internal mentation']
PATHS = ['Observation','Task_representation']


def geometry(x, correlation=True):
    x = np.array(x,dtype=float,copy=True)
    x -= x.mean(0)
    if correlation:
        x -= x.mean(1,keepdims=True)
        norm = np.linalg.norm(x,axis=1,keepdims=True)
        assert norm.min()>1e-12
        x /= norm
    return x


def transform_pca(x, params, correlation):
    x = np.asarray(x,float)-params['reference_center']
    if correlation:
        x -= x.mean(1,keepdims=True)
        norm = np.linalg.norm(x,axis=1,keepdims=True)
        assert norm.min()>1e-12
        x /= norm
    return (x-params['pca_mean'])@params['components'].T


def normalize(x,y):
    x = np.array(x,float,copy=True)
    centers = np.stack([x[y==c].mean(0) for c in [0,1]])
    x -= centers.mean(0)
    return x/np.sqrt(np.mean([np.mean(np.sum(x[y==c]**2,axis=1)) for c in [0,1]]))


def classical(x):
    x = x-x.mean(0)
    vals,vec = eigh(x@x.T,subset_by_index=[len(x)-3,len(x)-1])
    return vec[:,::-1]*np.sqrt(np.maximum(vals[::-1],0))


def orient(pair,y):
    high = pair[1]
    d = high[y==1].mean(0)-high[y==0].mean(0)
    d /= np.linalg.norm(d)
    _,_,v = np.linalg.svd(high-np.outer(high@d,d),full_matrices=False)
    t = v[0]-v[0].dot(d)*d
    t /= np.linalg.norm(t)
    old = np.column_stack([d,t,np.cross(d,t)])
    e,a = np.deg2rad([45,125])
    u = np.array([-np.sin(a),np.cos(a),0])
    w = np.array([-np.sin(e)*np.cos(a),-np.sin(e)*np.sin(a),np.cos(e)])
    rot = old@np.column_stack([u,w,np.cross(u,w)]).T
    np.testing.assert_allclose(rot.T@rot,np.eye(3),atol=1e-12)
    return [q@rot for q in pair]


def metrics(q,y):
    assert np.isfinite(q).all() and set(y)=={0,1}
    centers = np.stack([q[y==c].mean(0) for c in [0,1]])
    mid = centers.mean(0)
    total = np.mean([np.mean(np.sum((q[y==c]-mid)**2,axis=1)) for c in [0,1]])
    within = np.mean([np.mean(np.sum((q[y==c]-centers[c])**2,axis=1)) for c in [0,1]])
    w = np.sqrt(within/total)
    b = np.linalg.norm(centers[0]-centers[1])/np.sqrt(total)
    rho = spearmanr(pdist(q),pdist(y[:,None],'hamming')).statistic
    assert abs(w*w+b*b/4-1)<1e-10
    return np.array([w,b,rho])


def draw_indices(y, count, seed=None):
    pools = [np.flatnonzero(y==c) for c in [0,1]]
    rng = np.random.RandomState(seed) if seed is not None else None
    return np.array([np.concatenate([(rng if rng is not None else local).choice(p,len(p),replace=True)
        for p in pools]) for local in (np.random.RandomState(i) for i in range(count))],dtype=np.int16)


def fast(q,y,counts):

    n = len(y)
    ia,ib = np.triu_indices(n,1)
    dist = np.r_[0.,pdist(q)]
    order = np.argsort(dist,kind='stable')
    starts = np.r_[0,np.flatnonzero(np.diff(dist[order])!=0)+1]
    target = np.r_[0.,(y[ia]!=y[ib]).astype(float)][order]
    result = np.empty((len(counts),3))
    N = n*(n-1)/2
    midrank = (N+1)/2
    for start in range(0,len(counts),1024):
        c = counts[start:start+1024].astype(float)
        mus, seconds = [],[]
        for label in [0,1]:
            mask = y==label
            num = c[:,mask].sum(1)
            mus.append(c[:,mask]@q[mask]/num[:,None])
            seconds.append(c[:,mask]@(q[mask]**2).sum(1)/num)
        m = (mus[0]+mus[1])/2
        within = (seconds[0]-(mus[0]**2).sum(1)+seconds[1]-(mus[1]**2).sum(1))/2
        total = (seconds[0]+seconds[1])/2-(m**2).sum(1)
        w = np.column_stack([(c*(c-1)/2).sum(1),c[:,ia]*c[:,ib]])[:,order]
        g = np.add.reduceat(w,starts,axis=1)
        gy = np.add.reduceat(w*target,starts,axis=1)
        ranks = np.cumsum(g,axis=1)-(g-1)/2-midrank
        p = gy.sum(1)/N
        rho = (ranks*gy).sum(1)/np.sqrt((g*ranks**2).sum(1)*N*p*(1-p))
        result[start:start+len(c)] = np.column_stack([
            np.sqrt(np.maximum(within,0)/total),np.linalg.norm(mus[0]-mus[1],axis=1)/np.sqrt(total),rho])
    return result


def stars(p):
    for cutoff,label in [(1e-4,'****'),(1e-3,'***'),(.01,'**'),(.05,'*')]:
        if p<cutoff:
            return label
    return 'ns'
