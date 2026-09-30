
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def language_scores(rows,output,panel):
    grouped=defaultdict(list)
    for row in rows:grouped[row['subject']].append(float(row['BGEScore']))
    subjects=sorted(grouped)
    means=np.array([np.mean(grouped[s]) for s in subjects])
    with (output/'subject_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['subject','n_trials','BGEScore_mean']);w.writeheader()
        w.writerows(dict(subject=s,n_trials=len(grouped[s]),BGEScore_mean=float(v)) for s,v in zip(subjects,means))
    fig,ax=plt.subplots(figsize=(4,4))
    if len(means)>1:ax.boxplot(means,positions=[1],widths=.3,showfliers=False)
    ax.scatter(np.ones(len(means)),means,color='#356f95',s=30,zorder=3)
    ax.set(xticks=[1],xticklabels=['Ours'],ylabel='Mean BGE score per participant',title=f'Panel {panel}: {len(rows)} trials, {len(subjects)} participants')
    fig.tight_layout();fig.savefig(output/(panel+'.pdf'));fig.savefig(output/(panel+'.png'),dpi=200);plt.close(fig)

def image_scores(scores,output,panel):
    fig,ax=plt.subplots(figsize=(9,4));keys=list(scores)
    ax.bar(keys,[scores[k] for k in keys],color='#356f95')
    ax.tick_params(axis='x',rotation=35);ax.set(ylabel='Metric value',title=f'Panel {panel}: generated image evaluation')
    fig.tight_layout();fig.savefig(output/'image_metrics.pdf');fig.savefig(output/'image_metrics.png',dpi=200);plt.close(fig)
