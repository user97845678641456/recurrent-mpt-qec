"""Load committed circuits and the exact report fault vocabulary."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import scipy.sparse as sp
import stim
from .utils import dem_matrices,x_circuit,keys

ROOT=Path(__file__).resolve().parents[2]
POINTS=(.003,.004,.005,.006)
CODES=('bb72','bb90','bb144')

@lru_cache(maxsize=12)
def matrices(code,p=.003):
    if code not in CODES or p not in POINTS:raise ValueError('Supported: BB72/90/144 and p=.003/.004/.005/.006, 12 rounds')
    folder=ROOT/'assets'/code;meta=json.loads((folder/'metadata.json').read_text())
    a=np.load(folder/'code_matrices.npz');n=meta['n']
    b=SimpleNamespace(**{k:a[k] for k in a.files},n_half=n//2)
    c=stim.Circuit.from_file(folder/f'circuit_{p:.4f}.stim')
    hf,lf,pf=dem_matrices(c);h,l,q=dem_matrices(x_circuit(c,n))
    rows=np.arange(c.num_detectors).reshape(13,n)[:,n//2:].ravel()
    lookup={k:i for i,k in enumerate(keys(h,l))};projected=keys(hf[rows],lf)
    mapping=np.array([lookup.get(k,-1) for k in projected],dtype=np.int32)
    assert all(k==((),()) for k,j in zip(projected,mapping) if j<0)
    ok=mapping>=0;merged=-np.expm1(np.bincount(mapping[ok],weights=np.log1p(-2*pf[ok]),minlength=len(q)))/2
    assert np.allclose(merged,q,rtol=1e-10,atol=1e-12)
    return dict(b=b,c=c,meta=meta,hf=hf,lf=lf,pf=pf,h=h,l=l,q=q,rows=rows,mapping=mapping)

def test_data(code,p):
    a=np.load(ROOT/'data/report_test'/code/f'test_{p:.4f}.npz')
    dat=matrices(code,p)
    assert np.allclose(a['priors'],dat['q'],rtol=1e-10,atol=1e-12)
    truth=np.unpackbits(a['faults_packed'],axis=1,count=int(a['fault_count']),bitorder='little')
    assert np.array_equal((dat['h']@truth.T).T%2,a['detectors'][:,dat['rows']])
    assert np.array_equal((dat['l']@truth.T).T%2,a['labels'])
    return dict(detectors=a['detectors'],labels=a['labels'],truth=truth,seed=int(a['seed']))

def sample_dem(code,p,shots,seed):
    """Generate new Stim DEM samples; fault labels remain offline supervision."""
    dat=matrices(code,p);dem=dat['c'].detector_error_model().flattened()
    lookup={k:i for i,k in enumerate(keys(dat['h'],dat['l']))}
    rmap=np.full(dat['c'].num_detectors,-1,dtype=int);rmap[dat['rows']]=np.arange(len(dat['rows']))
    mapping=[]
    for op in dem:
        if op.type!='error':continue
        ds=set();ls=set()
        for t in op.targets_copy():
            if t.is_relative_detector_id():
                j=int(rmap[t.val])
                if j>=0:ds.symmetric_difference_update([j])
            elif t.is_logical_observable_id():ls.symmetric_difference_update([t.val])
        key=tuple(sorted(ds)),tuple(sorted(ls));assert key==((),()) or key in lookup;mapping.append(lookup.get(key,-1))
    mapping=np.asarray(mapping);ok=mapping>=0
    mt=sp.coo_matrix((np.ones(ok.sum(),np.uint8),(mapping[ok],np.flatnonzero(ok))),shape=(len(dat['q']),len(mapping))).tocsr()
    d,y,f=dem.compile_sampler(seed=seed).sample(shots,return_errors=True);g=(mt@f.astype(np.uint8).T).T%2
    assert np.array_equal((dat['h']@g.T).T%2,d[:,dat['rows']]) and np.array_equal((dat['l']@g.T).T%2,y)
    return dict(detectors=d,labels=y,truth=g,seed=seed)
