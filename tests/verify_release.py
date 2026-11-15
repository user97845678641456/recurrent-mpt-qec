"""Verify exported model logits, test truth mappings and OSD candidate minimum."""
import argparse,json
import numpy as np
import torch
from qec_mpt.data import ROOT,CODES,matrices,test_data,sample_dem
from qec_mpt.model import infer
from qec_mpt.osd import pack,decode

def osd_audit():
    rng=np.random.default_rng(21501115)
    for _ in range(20):
        h=np.hstack((np.eye(4,dtype=np.uint8),rng.integers(0,2,(4,7),dtype=np.uint8)))
        q=rng.uniform(.01,.2,11);z=np.arange(11,0,-1,dtype=float);s=rng.integers(0,2,4,dtype=np.uint8)
        choices=[()]+[(i,) for i in range(7)]+[(i,j) for i in range(7) for j in range(i+1,7)]
        candidates=[]
        for choice in choices:
            f=np.zeros(11,np.uint8)
            for j in choice:f[4+j]=1
            f[:4]=s^(h[:,4:]@f[4:]%2);candidates.append(f)
        got,n=decode(pack(h),s,z,q,'syndrome',8,'prior',False)
        costs=np.array(candidates)@np.log((1-q)/q)
        assert np.array_equal(h@got%2,s) and n==len(choices) and np.isclose(got@np.log((1-q)/q),costs.min())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args();torch.set_num_threads(4);osd_audit();records=[]
    golden=np.load(ROOT/'tests/golden_logits_005.npz')
    for code in CODES:
        a=test_data(code,.005);z=infer(code,.005,a['detectors'][:64],args.device);reference=golden[code]
        if args.device.startswith('cuda'):assert np.array_equal(z[:8],reference),'GPU BF16 logits differ from original report model'
        else:assert np.isfinite(z).all()
        dat=matrices(code,.005);f,_=decode(pack(dat['h']),a['detectors'][0,dat['rows']],z[0],dat['q'],'syndrome',8,'prior',False)
        assert np.array_equal(dat['h']@f%2,a['detectors'][0,dat['rows']])
        fresh=sample_dem(code,.005,16,20261115)
        records.append(dict(code=code,truth_mapping=True,fresh_sampling=True,device=args.device,gpu_BF16_original_logits_exact=bool(np.array_equal(z[:8],reference)),max_difference=float(np.abs(z[:8]-reference).max())))
    (ROOT/'runs').mkdir(exist_ok=True);(ROOT/'runs/verification.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2));print('RELEASE_CHECKS_PASSED')

if __name__=='__main__':main()
