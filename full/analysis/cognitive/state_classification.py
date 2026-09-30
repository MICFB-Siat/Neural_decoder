from pathlib import Path
import argparse,sys
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['E','F'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--subject');p.add_argument('--device',default='cpu');a=p.parse_args()
    sys.path.insert(0,str(ROOT/a.panel/'code'));import loso_state_runtime as runtime
    ds='ds000210' if a.panel=='E' else 'ds002835';version='full55_joint_mae_all_re_v1' if a.panel=='E' else 'ds002835_26_continual_mae_re_v1'
    rows=[]
    for method,folder in [('Ours','poe'),('NeuroStorm','neurostorm')]:
        report=runtime.run(ds,method,a.output/method,a.device,subject=a.subject,feature_root=ROOT/'inputs/state_features'/version/folder/ds,checkpoint_root=ROOT/a.panel/'weights')
        rows.extend(dict(method=method,**row) for row in report['per_subject'])
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();methods=['Ours','NeuroStorm'];ax.boxplot([frame[frame.method==m].accuracy*100 for m in methods],labels=methods);ax.set_ylabel('Accuracy (%)');fig.savefig(a.output/(a.panel+'.pdf'));fig.savefig(a.output/(a.panel+'.png'),dpi=200);plt.close(fig)
    print(frame)
if __name__=='__main__':main()
