"""Verify recorded numerical outputs and the experiment design."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    expected=json.loads((ROOT/'expected_checksums.json').read_text())
    for name,digest in expected.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=digest:raise AssertionError('Numerical output differs from reference: '+name)
    d=pd.read_csv(ROOT/'results/metrics.csv')
    assert len(d)==226800 and d.seed.nunique()==30 and d.dataset.nunique()==5
    for key,a in d.groupby(['dataset','model','seed','alpha']):
        assert a[a.method=='source_id'].threshold.nunique()==1
        p=a.pivot(index=['mechanism','ratio'],columns='method',values='threshold')
        assert (p.budget>=p.source_id).all()
    for p in (ROOT/'results/predictions').glob('*.npz'):
        a=np.load(p);tr,ca,te=[set(a[k]) for k in ['train_ids','cal_ids','test_ids']]
        assert not tr&ca and not tr&te and not ca&te
    print('Numerical fingerprints, split isolation, source invariance, and budget dominance verified.')
if __name__=='__main__':main()
