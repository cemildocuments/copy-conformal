"""Independent enumerations check the sharp frontier and conservative repair."""
import unittest,sys,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_experiments import calibrate,rank

def comps(total,n):
    if n==1:
        yield [total];return
    for b in range(total+1):
        for rest in comps(total-b,n-1):yield [b]+rest

class CalibrationTests(unittest.TestCase):
    def test_all_allocations(self):
        cases=0
        for g in range(2,8):
            s=np.arange(1,g+1,dtype=float)
            for b in range(7):
                for alpha in [.1,.2,.5]:
                    values=[]
                    clean=calibrate(s,np.ones(g,dtype=int),alpha,'source_id')
                    for extra in comps(b,g):
                        w=1+np.array(extra);expanded=np.sort(np.repeat(s,w));k=rank(g+b,alpha)
                        direct=np.inf if k>len(expanded) else expanded[k-1]
                        self.assertEqual(calibrate(s,w,alpha,'rowwise'),direct)
                        self.assertEqual(calibrate(s,w,alpha,'source_id'),clean)
                        self.assertGreaterEqual(calibrate(s,w,alpha,'budget'),clean)
                        values.append(direct);cases+=1
                    lower=np.inf if rank(g+b,alpha)>g+b else max(1,rank(g+b,alpha)-b)
                    self.assertEqual(min(values),lower)
        print('Exhaustive allocation cases:',cases)
    def test_unknown_budget_bound(self):
        rng=np.random.default_rng(842)
        for _ in range(1000):
            g=int(rng.integers(10,100));b=int(rng.integers(0,100));s=rng.normal(size=g);w=1+rng.multinomial(b,np.full(g,1/g))
            for bhat in [b,min(g+b-1,b+10)]:
                self.assertGreaterEqual(calibrate(s,w,.1,'budget',bhat),calibrate(s,w,.1,'source_id'))
    def test_ties(self):
        s=np.array([0.,0.,1.,1.,2.,2.,3.,3.,4.,4.]);w=np.array([20,1,2,1,1,1,1,1,1,1])
        self.assertGreaterEqual(calibrate(s,w,.1,'budget'),calibrate(s,w,.1,'source_id'))
    def test_infinity(self):
        s=np.arange(4.);w=np.array([100,1,1,1])
        self.assertTrue(np.isinf(calibrate(s,w,.1,'source_id')))
        self.assertTrue(np.isinf(calibrate(s,w,.1,'budget')))
        self.assertTrue(np.isfinite(calibrate(s,w,.1,'rowwise')))
if __name__=='__main__':unittest.main()
