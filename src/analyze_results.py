"""Recompute numerical summaries and budget sensitivity from saved scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_experiments import ROOT,weights_for,calibrate,SEED

def main():
 d=pd.read_csv(ROOT/'results/metrics.csv');a=d[d.alpha==.1]
 # Averages over models and tasks are descriptive; uncertainty is across split seeds.
 fields=['mechanism','ratio','method']
 grouped=a.groupby(fields+['seed'])[['coverage','width_ratio','finite']].mean().reset_index()
 out=grouped.groupby(fields).agg(coverage=('coverage','mean'),mc_se=('coverage',lambda x:x.std(ddof=1)/np.sqrt(len(x))),width_ratio=('width_ratio','mean'),finite=('finite','mean')).reset_index()
 out.to_csv(ROOT/'results/summary.csv',index=False)
 by=a[(a.ratio==4)&(a.mechanism=='low_score')].groupby(['dataset','method','seed']).coverage.mean().reset_index()
 by.groupby(['dataset','method']).coverage.agg(['mean','std']).reset_index().to_csv(ROOT/'results/dataset_summary.csv',index=False)
 p=grouped.pivot(index=['mechanism','ratio','seed'],columns='method',values='coverage').reset_index();p['difference']=p.source_id-p.rowwise
 p.groupby(['mechanism','ratio']).difference.agg(mean='mean',mc_se=lambda x:x.std(ddof=1)/np.sqrt(len(x))).reset_index().to_csv(ROOT/'results/paired.csv',index=False)
 rows=[]
 for name in ['airfoil','concrete','energy','wine_red','wine_white']:
  X=np.load(ROOT/'data/processed'/(name+'.npz'))['X']
  for model in ['ridge','forest','boosting']:
   for s in range(30):
    z=np.load(ROOT/'results/predictions'/f'{name}_{model}_{s:02d}.npz');cs=z['cal_score'];ts=z['test_score'];g=len(cs);b=4*g
    for mi,mech in [(2,'low_score'),(1,'random')]:
     rng=np.random.default_rng(np.random.SeedSequence([SEED,s,mi,5,197]));w=weights_for(cs,X[z['cal_ids'],0],b,mech,rng)
     base=calibrate(cs,w,.1,'source_id')
     for frac in [0,.25,.5,.75,1,1.1,1.2]:
      bh=min(int(round(frac*b)),g+b-1);q=calibrate(cs,w,.1,'budget',bh)
      rows.append(dict(dataset=name,model=model,seed=s,mechanism=mech,budget_ratio=frac,budget=bh,coverage=float(np.mean(ts<=q)),width_ratio=q/base,finite=int(np.isfinite(q))))
 pd.DataFrame(rows).to_csv(ROOT/'results/budget_sensitivity.csv',index=False)
 syn=pd.read_csv(ROOT/'results/synthetic.csv')
 invariant=a.groupby(['dataset','model','seed','method']).threshold.nunique()
 checks={'metric_rows':len(d),'source_id_invariant':bool((invariant.xs('source_id',level='method')==1).all()),'budget_dominates_source_id':bool((d[d.method=='budget'].threshold.to_numpy()>=d[d.method=='source_id'].threshold.to_numpy()).all()),'max_synthetic_abs_error':float(abs(syn.coverage-syn.theory).max()),'max_synthetic_standardized_error':float((abs(syn.coverage-syn.theory)/syn.mc_se.replace(0,np.nan)).max())}
 (ROOT/'results/checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks));print(out[(out.ratio==4)].to_string(index=False));print(pd.read_csv(ROOT/'results/paired.csv').query('mechanism=="low_score" and ratio==4').to_string(index=False))
if __name__=='__main__':main()
