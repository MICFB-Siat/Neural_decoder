
import argparse
import json
import numpy as np
import sklearn
from scipy.spatial.distance import pdist,squareform
from smacof_132 import _smacof_single
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    args=parser.parse_args()
    z=np.load(args.input)
    y=z['labels']
    coords,iterations=[],[]
    for key in ['low','high']:
        q,stress,n_iter=_smacof_single(squareform(pdist(z[key])),n_components=3,max_iter=10000,eps=1e-8,random_state=6,normalized_stress=False,init=np.random.RandomState(6).uniform(size=(len(y),3)))
        q-=np.stack([q[y==c].mean(0) for c in [0,1]]).mean(0)
        q/=np.sqrt(np.mean([np.mean(np.sum(q[y==c]**2,axis=1)) for c in [0,1]]))
        coords.append(q)
        iterations.append(n_iter)
    np.savez_compressed(args.output,coordinates=np.stack(coords),iterations=iterations,sklearn=sklearn.__version__)
    print(json.dumps(dict(sklearn=sklearn.__version__,iterations=iterations)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
