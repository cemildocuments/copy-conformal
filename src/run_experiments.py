"""Reproduce the copy-budget calibration study from fixed public data snapshots."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import argparse, json, time, sys, platform, hashlib, zipfile, io, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
RATIOS=[0,.25,.5,1,2,4,9]
MECHANISMS=['balanced','random','low_score','high_score','low_feature','minimum']
ALPHAS=[.05,.10,.20]
SEED=20261027

def rank(n,alpha): return int(math.ceil((n+1)*(1-alpha)-1e-12))
def order(scores,k,weights=None):
    if weights is None:
        if k>len(scores): return np.inf
        return float(np.partition(scores,k-1)[k-1])
    idx=np.argsort(scores,kind='stable');cum=np.cumsum(weights[idx])
    if k>cum[-1]: return np.inf
    return float(scores[idx[np.searchsorted(cum,k,side='left')]])

def calibrate(scores,weights,alpha,method,budget=None):
    g=len(scores);n=int(weights.sum())
    if method=='source_id': return order(scores,rank(g,alpha))
    if method=='rowwise': return order(scores,rank(n,alpha),weights)
    if method=='budget':
        b=n-g if budget is None else int(budget)
        if not 0<=b<n: raise ValueError('Copy upper bound must be between zero and N-1.')
        return order(scores,b+rank(n-b,alpha),weights)
    if method=='ess':
        neff=n*n/float(weights@weights)
        level=(1-alpha)*(1+1/neff)
        return np.inf if level>1 else order(scores,math.ceil(n*level-1e-12),weights)
    raise ValueError(method)

def weights_for(scores,features,b,mechanism,rng):
    g=len(scores);w=np.ones(g,dtype=np.int64)
    if b==0:return w
    m=max(1,math.ceil(g*.2))
    if mechanism=='balanced':
        w+=b//g;w[rng.permutation(g)[:b%g]]+=1
    elif mechanism=='random':w+=rng.multinomial(b,np.full(g,1/g))
    else:
        ids={'low_score':np.argsort(scores,kind='stable')[:m],
             'high_score':np.argsort(scores,kind='stable')[-m:],
             'low_feature':np.argsort(features,kind='stable')[:m],
             'minimum':np.array([np.argmin(scores)])}[mechanism]
        w[ids]+=rng.multinomial(b,np.full(len(ids),1/len(ids)))
    return w

def prepare():
    raw=ROOT/'data/raw';out=ROOT/'data/processed';out.mkdir(exist_ok=True)
    for rec in json.loads((ROOT/'data/download_manifest.json').read_text()):
        assert hashlib.sha256((raw/(rec['name']+'.zip')).read_bytes()).hexdigest()==rec['sha256']
    data={}
    with zipfile.ZipFile(raw/'airfoil.zip') as z:
        data['airfoil']=pd.read_csv(io.BytesIO(z.read('airfoil_self_noise.dat')),sep=r'\s+',header=None).to_numpy(float)
    with zipfile.ZipFile(raw/'concrete.zip') as z:
        name=next(n for n in z.namelist() if n.lower().endswith('.xls'))
        data['concrete']=pd.read_excel(io.BytesIO(z.read(name))).to_numpy(float)
    with zipfile.ZipFile(raw/'energy.zip') as z:
        a=pd.read_excel(io.BytesIO(z.read('ENB2012_data.xlsx'))).iloc[:,:10].dropna().to_numpy(float)
        data['energy']=a[:,:9]
    with zipfile.ZipFile(raw/'wine.zip') as z:
        for color in ['red','white']:data['wine_'+color]=pd.read_csv(io.BytesIO(z.read(f'winequality-{color}.csv')),sep=';').to_numpy(float)
    records=[]
    for name,a in data.items():
        assert np.isfinite(a).all()
        _,idx,counts=np.unique(a,axis=0,return_index=True,return_counts=True)
        ids=np.sort(idx);b=a[ids]
        np.savez_compressed(out/(name+'.npz'),X=b[:,:-1],y=b[:,-1],raw_source_row=ids)
        pd.DataFrame(b,columns=[f'x{i+1}' for i in range(a.shape[1]-1)]+['target']).to_csv(out/(name+'.csv'),index=False)
        records.append(dict(dataset=name,raw_rows=len(a),support_rows=len(b),features=a.shape[1]-1,removed=len(a)-len(b),train=int(.6*len(b)),cal=int(.2*len(b)),test=len(b)-int(.6*len(b))-int(.2*len(b))))
    pd.DataFrame(records).to_csv(ROOT/'results/datasets.csv',index=False)
    return list(data)

def task(spec):
    name,model_name,s=spec;t0=time.perf_counter();seed=SEED+s
    z=np.load(ROOT/'data/processed'/(name+'.npz'));X=z['X'];y=z['y']
    rng=np.random.default_rng(seed);ids=rng.permutation(len(y));nt=int(.6*len(y));nc=int(.2*len(y));tr,ca,te=ids[:nt],ids[nt:nt+nc],ids[nt+nc:]
    if model_name=='ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    elif model_name=='forest':model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features=1.0,random_state=seed,n_jobs=1)
    else:model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=seed)
    with threadpool_limits(limits=1):
        model.fit(X[tr],y[tr]);pc=model.predict(X[ca]);pt=model.predict(X[te])
    cs=np.abs(y[ca]-pc);ts=np.abs(y[te]-pt)
    np.savez_compressed(ROOT/'results/predictions'/f'{name}_{model_name}_{s:02d}.npz',train_ids=tr,cal_ids=ca,test_ids=te,cal_score=cs,test_score=ts,pred_cal=pc,pred_test=pt)
    rows=[]
    for mi,mech in enumerate(MECHANISMS):
        for ri,ratio in enumerate(RATIOS):
            b=int(round(ratio*nc));rr=np.random.default_rng(np.random.SeedSequence([SEED,s,mi,ri,197]))
            w=weights_for(cs,X[ca,0],b,mech,rr)
            for alpha in ALPHAS:
                clean=calibrate(cs,w,alpha,'source_id')
                for method in ['rowwise','source_id','budget','ess']:
                    q=calibrate(cs,w,alpha,method)
                    rows.append(dict(dataset=name,model=model_name,seed=s,mechanism=mech,ratio=ratio,alpha=alpha,method=method,G=nc,B=b,N=nc+b,coverage=float(np.mean(ts<=q)),width=2*q,width_ratio=q/clean,threshold=q,rmse=float(np.sqrt(np.mean((y[te]-pt)**2))),finite=int(np.isfinite(q))))
    return rows,dict(dataset=name,model=model_name,seed=s,seconds=time.perf_counter()-t0)

def synthetic():
    rows=[];reps=20000
    for di,dist in enumerate(['uniform','exponential']):
        for g in [20,50,100,250]:
            rng=np.random.default_rng(np.random.SeedSequence([SEED,di,g,739]));u=rng.random((reps,g+1))
            a=u if dist=='uniform' else -np.log1p(-u)
            ss=np.sort(a[:,:g],axis=1);test=a[:,-1]
            for alpha in ALPHAS:
                for ratio in RATIOS:
                    b=round(g*ratio);k=rank(g+b,alpha);r=max(1,k-b)
                    q=np.full(reps,np.inf) if k>g+b else ss[:,r-1]
                    f=np.ones(reps) if k>g+b else (q if dist=='uniform' else -np.expm1(-q))
                    theory=1 if k>g+b else r/(g+1)
                    rows.append(dict(distribution=dist,G=g,B=b,ratio=ratio,alpha=alpha,rank=r,trials=reps,theory=theory,coverage=float(np.mean(test<=q)),mc_se=float(np.std(test<=q,ddof=1)/np.sqrt(reps)),cdf_mean=float(f.mean()),cdf_mc_se=float(f.std(ddof=1)/np.sqrt(reps))))
    pd.DataFrame(rows).to_csv(ROOT/'results/synthetic.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);ap.add_argument('--seeds',type=int,default=30);args=ap.parse_args()
    (ROOT/'results/predictions').mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter();names=prepare();synthetic();tasks=[(d,m,s) for d in names for m in ['ridge','forest','boosting'] for s in range(args.seeds)]
    allrows=[];timings=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs=[ex.submit(task,t) for t in tasks]
        for i,f in enumerate(as_completed(fs)):
            rows,t=f.result();allrows.extend(rows);timings.append(t)
            if (i+1)%15==0:print(f'Completed {i+1}/{len(tasks)} fits in {time.perf_counter()-t0:.1f}s',flush=True)
    sort=['dataset','model','seed','mechanism','ratio','alpha','method']
    pd.DataFrame(allrows).sort_values(sort).to_csv(ROOT/'results/metrics.csv',index=False)
    pd.DataFrame(timings).sort_values(['dataset','model','seed']).to_csv(ROOT/'results/timings.csv',index=False)
    import sklearn,scipy
    meta=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,workers=args.workers,seeds=args.seeds,total_seconds=time.perf_counter()-t0,model_fits=len(tasks),config=dict(ratios=RATIOS,alphas=ALPHAS,mechanisms=MECHANISMS,seed=SEED))
    (ROOT/'results/environment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
