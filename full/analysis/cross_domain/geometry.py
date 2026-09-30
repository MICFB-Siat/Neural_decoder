from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
def centroid(x, labels):
    x = x.astype(np.float64)
    x = x - x.mean(0)
    x /= np.sqrt(np.mean(np.sum(x*x, axis=1)))
    return np.stack([x[labels == cell].mean(0) for cell in range(4)])

def metrics(c):
    c = np.asarray(c, dtype=np.float64)
    u, v = c[1] - c[0], c[3] - c[2]
    ps = float(u @ v / (np.linalg.norm(u)*np.linalg.norm(v)))
    singular = np.linalg.svd(c - c.mean(0), compute_uv=False)
    return dict(PS=ps, angle=float(np.degrees(np.arccos(np.clip(ps, -1, 1)))),
                depth=float(singular[2]**2 / np.sum(singular**2)))

def ccgp(x, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(0)
    cells = [np.flatnonzero(labels == i) for i in range(4)]
    n = min(map(len, cells))
    ys = np.r_[np.zeros(n), np.ones(n)]
    vals = []
    for _ in range(20):
        selected = [rng.permutation(ids)[:n] for ids in cells]
        for a, b in ((0, 2), (2, 0)):
            itrain, itest = np.r_[selected[a], selected[a+1]], np.r_[selected[b], selected[b+1]]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=.01, penalty='l2', max_iter=3000, random_state=0))
            vals.append(clf.fit(x[itrain], ys).score(x[itest], ys))
    return float(np.mean(vals))

def evaluate(domain, subject, low, high, y):
    groups=[np.flatnonzero(y==i) for i in range(4)]
    n=min(map(len,groups))
    if n<2: raise ValueError('Insufficient observations in a condition')
    if domain=='internal':
        return [(centroid(x,y),metrics(centroid(x,y))['PS'],ccgp(x,y)) for x in (low,high)]
    rng=np.random.default_rng(20260920 if domain=='perception' else 20260921+int(subject.split('-')[1]))
    draws=[[rng.choice(g,n,replace=False) for g in groups] for _ in range(200)]
    result=[]
    for x in (low,high):
        x=x.astype(float)
        if domain=='motor':
            x=x-x.mean(0);x/=np.sqrt(np.mean(np.sum(x*x,axis=1)))
        cs=[np.stack([x[ix].mean(0) for ix in d]) for d in draws]
        scores=[]
        for d in draws[:20]:
            for a,b in ((0,2),(2,0)):
                tr,te=np.r_[d[a],d[a+1]],np.r_[d[b],d[b+1]]
                kwargs={'C':.01,'max_iter':2000,'solver':'liblinear','random_state':0} if domain=='perception' else {'C':.01,'max_iter':3000}
                m=make_pipeline(StandardScaler(),LogisticRegression(**kwargs)).fit(x[tr],y[tr]%2)
                scores.append(balanced_accuracy_score(y[te]%2,m.predict(x[te])))
        result.append((np.mean(cs,axis=0),float(np.mean([metrics(c)['PS'] for c in cs])),float(np.mean(scores))))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--subjects',nargs='+');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);rows=[];centers={}
    for file in sorted((ROOT/'inputs/geometry').glob('*.npz')):
        domain,subject=file.stem.split('_',1)
        if a.subjects and subject not in a.subjects:continue
        with np.load(file) as z:values=evaluate(domain,subject,z['low'],z['high'],z['labels'])
        row={'domain':domain,'subject':subject}
        for level,(c,ps,cg) in zip(['low','high'],values):
            centers[f'{domain}|{subject}|{level}']=c;row['PS_'+level]=ps;row['CCGP_'+level]=cg
        rows.append(row)
    if not rows:raise ValueError('No input subjects')
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'metrics.csv',index=False)
    np.savez_compressed(a.output/'computed_centers.npz',**centers)
    domains=frame.domain.unique();fig,axes=plt.subplots(2,len(domains),figsize=(4*len(domains),6),squeeze=False)
    for j,domain in enumerate(domains):
        part=frame[frame.domain==domain]
        for i,m in enumerate(['PS','CCGP']):
            ax=axes[i,j];ax.boxplot([part[m+'_low'],part[m+'_high']],labels=['Low','High']);ax.set_title(domain);ax.set_ylabel(m)
    fig.tight_layout();fig.savefig(a.output/'B.pdf');fig.savefig(a.output/'B.png',dpi=200);plt.close(fig)
    representatives={'internal':'sub-0058','motor':'sub-108','perception':'sub-04'}
    fig=plt.figure(figsize=(12,7))
    for j,domain in enumerate(domains):
        sub=representatives[domain]
        if sub not in frame[frame.domain==domain].subject.values:sub=frame[frame.domain==domain].subject.iloc[0]
        for i,level in enumerate(['low','high']):
            c=centers[f'{domain}|{sub}|{level}'];u,s,v=np.linalg.svd(c-c.mean(0),full_matrices=False);q=u[:,:3]*s[:3]
            ax=fig.add_subplot(2,len(domains),i*len(domains)+j+1,projection='3d')
            for x,y,color in [(0,1,'#3668B4'),(2,3,'#E69235')]:
                ax.plot(*q[[x,y]].T,color=color);ax.scatter(*q[[x,y]].T,color=color)
            ax.set_title(domain+' '+level+' '+sub)
    fig.tight_layout();fig.savefig(a.output/'A.pdf');fig.savefig(a.output/'A.png',dpi=200);plt.close(fig)
    print(frame.to_string(index=False))

if __name__=='__main__':main()
