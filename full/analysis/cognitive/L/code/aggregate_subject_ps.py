from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
d=pd.read_csv(P/'PS_pairs.csv');rows=[]
for (emotion,method),g in d.groupby(['emotion','method']):
 for subject in sorted(set(g.subject_i)|set(g.subject_j)):
  v=g[(g.subject_i==subject)|(g.subject_j==subject)].PS
  rows.append(dict(emotion=emotion,method=method,subject=subject,mean_PS_to_other_subjects=v.mean(),n_other_subjects=len(v)))
t=pd.DataFrame(rows);t.to_csv(P/'PS_subject_mean.csv',index=False)
t[t.emotion=='happy'].to_csv(P/'panel_c_plot_values.csv',index=False)
