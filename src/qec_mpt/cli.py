"""Inference and paired reproduction with both BP baselines."""
import argparse,json,concurrent.futures as cf
from pathlib import Path
import numpy as np
from .data import ROOT,CODES,POINTS,matrices,test_data,sample_dem

def baseline_job(arg):
    code,p,d,y,start=arg
    from ldpc import BpDecoder
    from .osd import pack,decode
    dat=matrices(code,p);h,l,q=dat['h'],dat['l'],dat['q'];pk=pack(h);out={}
    for method in ['minimum_sum','product_sum']:
        bp=BpDecoder(h,error_channel=q.tolist(),max_iter=200,bp_method=method,ms_scaling_factor=0.,schedule='parallel',input_vector_type='syndrome',omp_thread_count=1)
        bad=[];nan=inf=0
        for syn,label in zip(d[:,dat['rows']],y):
            bp.decode(syn.astype(np.uint8));z=-np.array(bp.log_prob_ratios,copy=True);nan+=int(np.isnan(z).sum());inf+=int(np.isinf(z).sum())
            z=np.where(np.isnan(z),np.log(q/(1-q)),z);z=np.nan_to_num(z,posinf=1e6,neginf=-1e6)
            f,_=decode(pk,syn,z,q,'syndrome',8,'prior',False);assert np.array_equal(h@f%2,syn);bad.append(np.any(l@f%2!=label))
        out[method]=np.asarray(bad);out[method+'_nan']=nan;out[method+'_inf']=inf
    return start,out

def evaluate(args):
    import torch
    from .model import infer
    from .osd import pack,decode
    torch.set_num_threads(4);folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True);summary=[]
    codes=CODES if args.code=='all' else [args.code];points=POINTS if args.noise=='all' else [float(args.noise)]
    for code in codes:
        for p in points:
            a=test_data(code,p) if args.dataset=='report' else sample_dem(code,p,args.shots,args.seed)
            n=len(a['labels']) if args.limit==0 else min(args.limit,len(a['labels']));d=a['detectors'][:n];y=a['labels'][:n]
            dat=matrices(code,p);h,l,q=dat['h'],dat['l'],dat['q'];pk=pack(h)
            z=infer(code,p,d,args.device);bad=[]
            for syn,zz,label in zip(d[:,dat['rows']],z,y):
                f,_=decode(pk,syn,zz,q,'syndrome',8,'prior',False);assert np.array_equal(h@f%2,syn);bad.append(np.any(l@f%2!=label))
            jobs=[(code,p,d[i:i+100],y[i:i+100],i) for i in range(0,n,100)]
            with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:parts=sorted(pool.map(baseline_job,jobs),key=lambda v:v[0])
            values={'MPT_CS8':np.array(bad),**{method:np.concatenate([v[1][method] for v in parts]) for method in ['minimum_sum','product_sum']}}
            dest=folder/f'{code}_{p:.4f}.npz';np.savez_compressed(dest,**values)
            row=dict(code=code,p=p,shots=n,error_rates={k:float(v.mean()) for k,v in values.items()},failures={k:int(v.sum()) for k,v in values.items()})
            if args.dataset=='report' and n==len(a['labels']):
                ms=json.loads((ROOT/'results/min_sum/results.json').read_text());spa=json.loads((ROOT/'results/sum_product/results.json').read_text())
                r=next(r for r in ms if r['code']==code and r['p']==p);s=next(r for r in spa if r['code']==code and r['p']==p)
                expected={'MPT_CS8':r['arms']['MPT_CS8_prior']['failures'],'minimum_sum':r['arms']['BP_CS8_prior']['failures'],'product_sum':s['arms']['SPA_CS8']['failures']}
                row['expected_report_failures']=expected;row['matches_report']=row['failures']==expected
            summary.append(row);print(json.dumps(row),flush=True);(folder/'summary.json').write_text(json.dumps(summary,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--code',choices=['all',*CODES],default='all');p.add_argument('--noise',choices=['all',*[str(v) for v in POINTS]],default='all');p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--output',default='runs/evaluation');p.add_argument('--dataset',choices=['report','fresh'],default='report');p.add_argument('--shots',type=int,default=3000);p.add_argument('--seed',type=int,default=20261115)
    p=sub.add_parser('train');p.add_argument('--code',choices=CODES,required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=1000000);p.add_argument('--initialize');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=21501115);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--schedule',choices=['cosine','step'],default='step');p.add_argument('--batch',type=int,default=64);p.add_argument('--validation-shots',type=int,default=1024);p.add_argument('--eval-every',type=int,default=100000)
    args=parser.parse_args()
    if args.command=='evaluate':evaluate(args)
    else:
        import torch
        from .training import train
        torch.set_num_threads(4);train(args)

if __name__=='__main__':main()
