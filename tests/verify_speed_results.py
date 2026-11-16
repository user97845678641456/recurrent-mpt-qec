"""Audit timings, model hashes and preserved predictions without rerunning clocks."""
import hashlib,json
import numpy as np
from qec_mpt.data import ROOT

def main():
    folder=ROOT/'results/speed';data=json.loads((folder/'results.json').read_text());protocol=data['protocol']
    verified=0
    for code_result in data['results']:
        code=code_result['code'];path=ROOT/'checkpoints'/f'{code}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['checkpoint_hashes'][code]
        test=np.load(ROOT/'data/report_test'/code/'test_0.0050.npz')
        reference=np.load(ROOT/'results/sum_product'/code/'predictions_0.0050.npz')
        assert len(code_result['batch'])==len(protocol['workers'])*len(protocol['batch_sizes'])
        for row in code_result['batch']:
            b=row['batch_size'];a=np.load(folder/f"{code}_w{row['workers']}_b{b}.npz");ix=a['dataset_indices']
            assert len(ix)==protocol['batch_shots']*protocol['repeats']
            assert len(np.unique(ix))==protocol['batch_shots']
            for method,arm in [('BP','SPA_CS8'),('MPT','MPT_CS8')]:
                seconds=a[method+'_seconds'];r=row['methods'][method]
                assert len(seconds)==protocol['batch_shots']//b*protocol['repeats'] and np.all(seconds>0)
                assert np.isclose(seconds.mean()*1000/b,r['total']['amortized_ms_per_frame'])
                assert np.isclose(b/seconds.mean(),r['total']['frames_per_second'])
                pred=a[method+'_logical_predictions'];agree=np.mean(np.all(pred==reference[arm+'_prediction'][ix],axis=1))
                assert np.isclose(agree,r['reference_logical_prediction_agreement'])
                assert np.isclose(np.mean(np.any(pred!=test['labels'][ix],axis=1)),r['sample_FER'])
                if method=='BP' or b==64:assert agree==1.
            assert np.allclose(a['MPT_frontend_seconds']+a['MPT_post_seconds'],a['MPT_seconds'])
            verified+=1
        primary=next(v for v in code_result['batch'] if v['batch_size']==64 and v['workers']==32)
        raw=np.load(folder/f'{code}_w32_b64.npz');n=protocol['batch_shots']//64
        ratios=raw['BP_seconds'].reshape(protocol['repeats'],n).sum(1)/raw['MPT_seconds'].reshape(protocol['repeats'],n).sum(1)
        print(code,'primary throughput ratio',primary['BP_over_MPT_speed_ratio'],'per-repeat ratios',ratios.tolist())
    assert verified==36
    print('SPEED_RESULT_AUDIT_PASSED',verified)

if __name__=='__main__':main()
