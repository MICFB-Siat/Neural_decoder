import functools
import numpy as np
import nibabel as nib
from scipy import sparse
from common.runtime import TEMPLATES

@functools.lru_cache(None)
def cortex_mask(hemi):
    return np.loadtxt(TEMPLATES/f'{hemi}_mask.txt')!=0

@functools.lru_cache(None)
def surface(hemi):
    g=nib.load(TEMPLATES/f'{hemi}_midthickness.gii')
    return np.asarray(g.darrays[0].data,dtype=np.float64),np.asarray(g.darrays[1].data,dtype=np.int64)

@functools.lru_cache(None)
def adjacency(hemi):
    _,f=surface(hemi)
    r=np.concatenate([f[:,0],f[:,1],f[:,2],f[:,1],f[:,2],f[:,0]])
    c=np.concatenate([f[:,1],f[:,2],f[:,0],f[:,0],f[:,1],f[:,2]])
    a=sparse.coo_matrix((np.ones(len(r)),(r,c)),shape=(32492,32492)).tocsr()
    a.data[:]=1
    degree=np.asarray(a.sum(1)).ravel();degree[degree==0]=1
    return a,degree

def smooth_surface(hemi,value,iters=22):
    a,d=adjacency(hemi);v=np.nan_to_num(value)
    for _ in range(iters):v=(a@v)/d
    return v

def basis(hemi):
    return np.load(TEMPLATES/f'{hemi}_basis.npy',mmap_mode='r')
