
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import internal_high_scan_20260923 as stats
import redraw_original_3d_20260923 as drawing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
OUT=BASE/'v2/final_low_layer_search_20260923'
REP={'internal':'sub-0058','motor':'sub-108'}
ARMS=[('lr3e5_wd1e4',3e-5,1e-4),('lr1e4_wd1e4',1e-4,1e-4),('lr1e4_wd1e2',1e-4,1e-2)]



drawing.BLUE,drawing.ORANGE='#E69235','#3668B4'


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)


def status(domain,phase,**kw):
    write_json(OUT/domain/'status.json',dict(phase=phase,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))


def load_fixed(domain):
    if domain=='internal':
        a=np.load(BASE/'v2/internal_high_scan_20260923/fixed_analysis_data.npz')
        h=np.load(BASE/'v2/internal_high_scan_20260923/features/epoch_012.npz')['obs_mu']
        hc=np.load(BASE/'v2/internal_high_scan_20260923/centroids/epoch_012__obs_mu.npz')
        hm=pd.read_csv(BASE/'v2/internal_high_scan_20260923/metrics/epoch_012__obs_mu.csv').set_index('subject')
        return {str(a['subjects'][i]):dict(low=a['low'][i],high=h[i],cells=a['cells'][i],high_c=hc[str(a['subjects'][i])],
                high_PS=float(hm.loc[str(a['subjects'][i]),'high_PS']),high_CCGP=float(hm.loc[str(a['subjects'][i]),'high_CCGP']),
                trials=a['selected_trials'][i],videos=a['videos']) for i in np.flatnonzero(a['eligible'])}
    meta=pd.read_csv(REPO/'figures/compositional_geometry_experiment/three_domains_real/motor_trial_context.csv')
    hm=pd.read_csv(BASE/'v2/motor_high_scan_20260921/all_metrics.csv').query("candidate=='seed1_mu_mean'").set_index('subject')
    hc=np.load(BASE/'v2/motor_high_scan_20260921/centroids.npz')
    result={}
    for sub,r in hm.iterrows():
        part=meta[meta.subject==sub];take=part.run.isin(['MIpre','MIpost','eegNF']).to_numpy()
        cells=part.label.to_numpy()[take]+2*part.run.eq('eegNF').to_numpy()[take]
        low=np.load(BASE/f'v2/lioi_pretrained/{sub}.npz')
        high=np.load(BASE/f'v2/motor_high_scan_20260921/features/seed1_{sub}.npz')
        assert np.array_equal(part.label.to_numpy(),low['y_true']) and np.array_equal(low['y_true'],high['y_true'])
        idx=[np.flatnonzero(cells==c) for c in range(4)];n=min(map(len,idx))
        rng=np.random.default_rng(20260921+int(sub.split('-')[1]))
        draws=[[rng.choice(ids,n,replace=False) for ids in idx] for _ in range(200)]
        result[sub]=dict(low=low['o'][take],high=high['mu_mean'][take],cells=cells,take=take,draws=draws,
                        high_c=hc[f'seed1_mu_mean|{sub}'],high_PS=float(r.ps_high_mean200),high_CCGP=float(r.ccgp_high))
    return result


def centers(x,y,draws=None):
    x=x.astype(float)-x.mean(0)
    x/=np.sqrt(np.mean(np.sum(x*x,axis=1)))
    if draws is None:
        c=np.stack([x[y==i].mean(0) for i in range(4)])
        return c,stats.metrics(c)['PS']
    allc=[np.stack([x[ix].mean(0) for ix in draw]) for draw in draws]
    return np.mean(allc,axis=0),float(np.mean([stats.metrics(c)['PS'] for c in allc]))


def motor_ccgp(x,draws):
    vals=[]
    for sel in draws[:20]:
        n=len(sel[0]);target=np.r_[np.zeros(n),np.ones(n)]
        for a,b in ((0,2),(2,0)):
            tr,te=np.r_[sel[a],sel[a+1]],np.r_[sel[b],sel[b+1]]
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000))
            vals.append(model.fit(x[tr],target).score(x[te],target))
    return float(np.mean(vals))


def evaluate(domain,name,features,metadata):
    dest=OUT/domain
    for d in ('features','centroids','metrics','figures','metadata'):(dest/d).mkdir(parents=True,exist_ok=True)
    fixed=load_fixed(domain)
    assert set(features)==set(fixed)
    np.savez_compressed(dest/f'features/{name}.npz',**features)
    rows=[];cent={}
    for sub,f in fixed.items():
        x=np.asarray(features[sub],float)
        assert x.shape[0]==len(f['cells']) and np.isfinite(x).all()
        assert np.std(x)>1e-9
        c,ps=centers(x,f['cells'],f.get('draws'))
        cc=(motor_ccgp(x,f['draws']) if domain=='motor' else stats.ccgp(x,f['cells']))
        lm,hm=stats.metrics(c),stats.metrics(f['high_c'])
        rows.append(dict(domain=domain,candidate=name,subject=sub,PS_low=ps,PS_high=f['high_PS'],
            CCGP_low=cc,CCGP_high=f['high_CCGP'],PS_low_display=lm['PS'],PS_high_display=hm['PS'],
            angle_low=lm['angle'],angle_high=hm['angle'],depth_low=lm['depth'],depth_high=hm['depth'],
            low_distance_to_90=abs(lm['angle']-90),layer=metadata.get('layer','baseline'),
            arm=metadata.get('arm','baseline'),epoch=metadata.get('epoch',0)))
        cent[f'{sub}|low']=c;cent[f'{sub}|high']=f['high_c']
    frame=pd.DataFrame(rows)
    frame.to_csv(dest/f'metrics/{name}.csv',index=False)
    np.savez_compressed(dest/f'centroids/{name}.npz',**cent)
    summary=[dict(domain=domain,candidate=name,metric=m,**stats.paired(frame[f'{m}_low'],frame[f'{m}_high'])) for m in ('PS','CCGP')]
    write_json(dest/f'metadata/{name}.json',dict(**metadata,statistics=summary,
        high_fixed='epoch12_obs_mu' if domain=='internal' else 'seed1_mu_mean',
        point_unit='participant; mean resampling PS for Motor, fixed-cell PS for Internal'))
    sub=REP[domain]
    drawing.draw_pair('Internal mentation' if domain=='internal' else 'Motor',sub,cent[f'{sub}|low'],cent[f'{sub}|high'],
        'fixed epoch12_obs_mu' if domain=='internal' else 'fixed seed1_mu_mean','standard',dest/f'figures/{name}.png')
    r=frame.set_index('subject').loc[sub]
    print(domain,name,'representative angle',round(r.angle_low,2),'->',round(r.angle_high,2),flush=True)
    return summary


def aggregate(domain):
    dest=OUT/domain
    files=sorted((dest/'metrics').glob('*.csv'))
    if not files:return
    allp=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    allp.to_csv(dest/'participant_metrics.csv',index=False)
    rows=[]
    for f in sorted((dest/'metadata').glob('*.json')):rows+=json.loads(f.read_text())['statistics']
    group=pd.DataFrame(rows)
    for m in ('PS','CCGP'):
        mask=group.metric==m
        for test in ('wilcoxon','signflip'):
            group.loc[mask,f'{test}_q_BH']=stats.bh(group.loc[mask,f'{test}_p'])
    group.to_csv(dest/'group_summary.csv',index=False)
    rep=allp[allp.subject==REP[domain]].sort_values('low_distance_to_90')
    rep.to_csv(dest/'ranked_representative_candidates.csv',index=False)
    html=['<!doctype html><meta charset="utf-8"><title>低层扫描</title>',
          '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}img{max-width:480px}table{font-size:12px}td{padding:4px}</style>',
          f'<h1>{domain}：低层候选，高层固定</h1><p>按真实低层夹角距90°排序。不是按屏幕投影排序；候选属于探索性筛选，全部指标和不显著结果均保留。蓝色为原上方情境箭头。</p>',
          '<p><a href="group_summary.csv">群体PS/CCGP与配对检验</a> · <a href="participant_metrics.csv">全体被试指标</a></p>',rep.to_html(index=False)]
    for r in rep.head(15).itertuples():
        html.append(f'<h2>{r.candidate}: {r.angle_low:.1f}° → {r.angle_high:.1f}°</h2><a href="figures/{r.candidate}.png"><img src="figures/{r.candidate}.png"></a>')
    (dest/'index.html').write_text('\n'.join(html),encoding='utf-8')


def root_index():
    text=['<!doctype html><meta charset="utf-8"><title>低层扫描总览</title><style>body{font:18px sans-serif;max-width:1100px;margin:40px auto}pre{white-space:pre-wrap}</style>',
          '<h1>低层扫描：高层固定</h1><p>Internal：sub-0058 / epoch12 obs_mu；Motor：sub-108 / seed1_mu_mean。原3D画法，原上方橙色箭头改蓝色、另一情境改橙色。所有低层训练仅用自监督目标，状态/评分标签不进入训练损失。</p>']
    for d in ('internal','motor'):
        s=OUT/d/'status.json'
        text.append(f'<h2>{d}</h2>')
        if (OUT/d/'index.html').exists():text.append(f'<a href="{d}/index.html">打开已完成候选</a>')
        if s.exists():text.append('<pre>'+s.read_text()+'</pre>')
    text.append('<p>状态与图表会随后台计算更新；浏览器刷新查看。</p>')
    if (OUT/'group_plot_data/paired_statistics.csv').exists():
        text.append('<h2>三域群体绘图数据</h2><p><a href="group_plot_data/paired_statistics.csv">配对统计</a> · <a href="group_plot_data/PS_per_participant.csv">PS</a> · <a href="group_plot_data/CCGP_per_participant.csv">CCGP</a></p><p>候选选取与Perception版本限定见导出说明；星号由实际数据计算。</p>')
    if (OUT/'final_selected/selection.json').exists():
        text.append('<h2>最终三域图</h2><p><a href="final_selected/internal.png">Internal</a> · <a href="final_selected/motor.png">Motor</a> · <a href="final_selected/perception.png">Perception</a> · <a href="final_selected/selection.json">选择与固定约束</a></p>')
    if (OUT/'recommendations_15/index.html').exists():
        text.append('<h2>Motor / Internal 各15个完整标注推荐</h2><p><a href="recommendations_15/index.html">打开推荐总入口</a> · <a href="ALL_IMAGE_PATHS.csv">全部图片路径CSV</a> · <a href="ALL_IMAGE_PATHS.txt">全部图片路径TXT</a></p>')
    if (OUT/'motor_bold_search_20260924/index.html').exists():
        text.append('<h2>Motor大胆参数搜索</h2><p><a href="motor_bold_search_20260924/index.html">打开89.98°候选及验证结果</a></p>')
    tmp=OUT/'index.tmp.html';tmp.write_text('\n'.join(text),encoding='utf-8');tmp.replace(OUT/'index.html')
