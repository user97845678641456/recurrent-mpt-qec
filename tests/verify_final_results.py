"""Check every released test set/result count and replay neural CS8 prefixes."""
import json,hashlib
import numpy as np
import torch
from qec_mpt.data import ROOT,CODES,POINTS,matrices,test_data
from qec_mpt.model import infer
from qec_mpt.osd import pack,decode

def main():
    torch.set_num_threads(4)
    rows=json.loads((ROOT/'results/sum_product/results.json').read_text())
    protocol=json.loads((ROOT/'provenance/protocol.json').read_text())
    records=[]
    for code in CODES:
        path=ROOT/'checkpoints'/f'{code}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['checkpoints'][code]['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['training_loss']=='fault_BCE_only'
        assert not any(k=='query' or k.startswith(('head.','read.','qnorm.')) for k in state['parameters'])
        for p in POINTS:
            a=test_data(code,p);dat=matrices(code,p);h,l,q=dat['h'],dat['l'],dat['q']
            r=next(r for r in rows if r['code']==code and r['p']==p)
            stored=np.load(ROOT/'results/sum_product'/code/f'predictions_{p:.4f}.npz')
            assert len(a['labels'])==r['shots']
            for arm in ['MPT_CS8','MS_CS8','SPA_CS8']:
                bad=np.any(stored[arm+'_prediction']!=a['labels'],axis=1)
                assert np.array_equal(bad,stored[arm+'_bad'])
                assert int(bad.sum())==r['arms'][arm]['failures']
            z=infer(code,p,a['detectors'][:64],'cuda');pk=pack(h)
            for i in range(8):
                syn=a['detectors'][i,dat['rows']];f,_=decode(pk,syn,z[i],q,'syndrome',8,'prior',False)
                assert np.array_equal(h@f%2,syn)
                assert np.array_equal(l@f%2,stored['MPT_CS8_prediction'][i])
            records.append(dict(code=code,p=p,shots=r['shots'],truth_mapping=True,all_result_counts_match=True,replayed_neural_CS8_frames=8))
            print('VERIFIED',code,p,flush=True)
    assert sum(r['shots'] for r in records)==32000
    (ROOT/'provenance/final_verification.json').write_text(json.dumps(records,indent=2))
    print('FINAL_RESULT_VERIFICATION_PASSED',flush=True)

if __name__=='__main__':main()
