
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl_original_3d_redraw')
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import ast
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BASE=Path(__file__).resolve().parent
SRC=BASE/'v2/internal_high_scan_20260923'
OUT=BASE/'v2/internal_motor_original_3d_redraw_20260923'
CAMERAS={'standard':(28,-60),'depth':(22,-115),'elevated':(48,-68)}
BLUE,ORANGE,EDGE='#3668B4','#E69235','#C9CDD3'
LIGHT=np.array([.3,-.5,.8]);LIGHT/=np.linalg.norm(LIGHT)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})

original=BASE/'motor_internal_3d_candidates_20260923.py'
source=original.read_text()
names={'ps','full_metrics','exact3','shade','panel','draw_pair'}
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
functions='\n\n'.join(ast.get_source_segment(source,n) for n in nodes)
functions=functions.replace('(0, 1, ORANGE), (2, 3, BLUE)','(0, 1, BLUE), (2, 3, ORANGE)')
functions=functions.replace('lim = np.abs(p).max() * 1.20','lim = np.abs(p).max() * 1.08')
functions=functions.replace('fig = plt.figure(figsize=(5.3, 8.0))','fig = plt.figure(figsize=(6.2, 8.0))')
functions=functions.replace(
    'if domain == "Motor":\n        desc = "rest → right-hand motor imagery; no feedback / EEG neurofeedback"\n    else:\n        desc = "engagement low → high; low / high sadness"',
    'if domain == "Motor":\n        desc = "rest → right-hand motor imagery; no feedback / EEG neurofeedback"\n    elif domain == "Perception":\n        desc = "inanimate → animate; small / large real-world size"\n    else:\n        desc = "engagement low → high; low / high sadness"')
functions=functions.replace('fig.savefig(destination, dpi=180)',
    'fig.savefig(destination, dpi=180)\n    fig.savefig(destination.with_suffix(".pdf"))')
exec(compile(functions,str(original),'exec'),globals())


def contact(candidate):
    lo=np.load(SRC/'centroids/low.npz')
    hi=np.load(SRC/f'centroids/{candidate}.npz')
    fig=plt.figure(figsize=(18,34))
    grid=fig.add_gridspec(8,4,left=.03,right=.97,bottom=.035,top=.955,wspace=.12,hspace=.30)
    for i,sub in enumerate(sorted(lo.files)):
        row,col=divmod(i,4)
        for k,(data,label) in enumerate(((lo,'Low'),(hi,'High'))):
            panel(fig.add_subplot(grid[2*row+k,col],projection='3d'),data[sub],f'{sub} | {label}',CAMERAS['standard'])
    control=' | RANDOM HIGH CONTROL' if candidate.startswith('epoch_000') else ''
    fig.suptitle(f'Internal mentation | {candidate}{control}\nEngagement x sadness | transductive | all 16 participants\nOriginal cubic axes, perspective projection and shaded surfaces; blue: low sadness, orange: high sadness',fontsize=16,y=.992)
    fig.savefig(OUT/f'all_subjects/{candidate}.png',dpi=95)
    plt.close(fig)
    return candidate


def main():
    for folder in ('selected','all_subjects','motor','data','code'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    shortlist=pd.read_csv(SRC/'shortlist.csv')
    low=np.load(SRC/'centroids/low.npz')
    checks=[]
    html=['<!doctype html><meta charset="utf-8"><title>原方案3D重绘</title>',
        '<style>body{font:16px sans-serif;max-width:1450px;margin:30px auto;color:#263447}img{width:32%;vertical-align:top}a{color:#3668B4}</style>',
        '<h1>原方案3D重绘</h1><p>恢复原立方体坐标框、透视投影、半透明面、坐标网格、原视角与上下布局。保留指定蓝橙配色。数据、PS、CCGP及候选排序不变。</p>',
        '<p>三视角依次为standard、depth、elevated；上下使用相同视角。Internal仍为transductive。每个点为条件中心。</p>',
        '<p><a href="data/group_summary.csv">群体统计</a> · <a href="data/centroids_plot3d.csv">三维坐标</a> · <a href="data/shortlist.csv">候选列表</a></p>']
    for r in shortlist.itertuples():
        high=np.load(SRC/f'centroids/{r.candidate}.npz')[r.subject]
        label='原融合PoE' if r.layer=='poe_mu' else '监督专家分支（非融合PoE）'
        status='PS/CCGP通过BH校正' if r.both_metrics_BH_significant else '仅原始P达标，未全部通过BH校正'
        html.append(f'<h2>{r.subject} · {r.candidate}</h2><p>{label}；{status}。夹角 {r.low_angle:.1f}° → {r.high_angle:.1f}°。</p>')
        for view in CAMERAS:
            rel=f'selected/{r.candidate}/{r.subject}_{view}.png'
            de,pe=draw_pair('Internal mentation',r.subject,low[r.subject],high,r.candidate,view,OUT/rel)
            checks.append(dict(candidate=r.candidate,subject=r.subject,view=view,distance_error=de,PS_error=pe))
            html.append(f'<a href="{rel}"><img loading="lazy" src="{rel}"></a>')
        print('Redrawn',r.candidate,r.subject,flush=True)
    saved=np.load(BASE/'v2/selected_motor_sub108_mu_mean_3d_20260923/data/selected_centroids.npz')
    html.append('<h2>Motor：已保留sub-108 / seed1_mu_mean</h2>')
    for view in CAMERAS:
        rel=f'motor/sub-108_{view}.png'
        de,pe=draw_pair('Motor','sub-108',saved['low'].astype(float),saved['high'].astype(float),'seed1_mu_mean',view,OUT/rel)
        checks.append(dict(candidate='motor_seed1_mu_mean',subject='sub-108',view=view,distance_error=de,PS_error=pe))
        html.append(f'<a href="{rel}"><img src="{rel}"></a>')
    candidates=pd.read_csv(SRC/'group_summary.csv').candidate.unique()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,c in enumerate(pool.map(contact,candidates)):
            if (i+1)%20==0:print('All-subject galleries',i+1,'/',len(candidates),flush=True)
    html.append('<h2>全部配置：16名被试原方案3D图</h2>')
    for c in candidates:html.append(f'<p><a href="all_subjects/{c}.png">{c}</a></p>')
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    for name in ('group_summary.csv','participant_metrics.csv','shortlist.csv','centroids_plot3d.csv'):
        shutil.copy2(SRC/name,OUT/'data'/name)
    frame=pd.DataFrame(checks)
    assert frame.distance_error.max()<1e-10 and frame.PS_error.max()<1e-9
    frame.to_csv(OUT/'geometry_checks.csv',index=False)
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    shutil.copy2(original,OUT/'code'/original.name)
    (OUT/'code/original_functions_with_palette.py').write_text(functions)
    manifest={'source_data':str(SRC),'original_drawing_source':str(original),'changes':['blue/orange palette only','also export PDF'],
              'original_style_restored':['cubic symmetric axes, margin 1.20','default perspective','shaded triangular surfaces','MDS labels and grid','5.3 x 8 inch pair layout'],
              'cameras':CAMERAS,'internal_pairs':len(shortlist),'all_subject_galleries':len(candidates),
              'max_distance_error':float(frame.distance_error.max()),'max_PS_error':float(frame.PS_error.max()),
              'data_retrained':False,'old_outputs_overwritten':False}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pass
    (OUT/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n'
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
