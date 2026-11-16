"""Warm end-to-end latency/throughput with independent CPU-parallel BP and CS8.

Run from an installed checkout. Worker processes import no neural model and
retain their own BP instance. Both methods share the same CPU pool and CS8;
all host/device transfers and task/result IPC are inside the batch clock.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMEXPR_NUM_THREADS']='1'
import argparse,concurrent.futures as cf,hashlib,json,multiprocessing as mp
import platform,time
from pathlib import Path
import numpy as np
from ldpc import BpDecoder
from qec_mpt.osd import decode,pack

def initialize(h,q,pk,warm_syndrome,barrier=None):
    global H,Q,PK,BP
    H,Q,PK=h,q,pk
    BP=BpDecoder(h,error_channel=q.tolist(),max_iter=200,bp_method='product_sum',schedule='parallel',input_vector_type='syndrome',omp_thread_count=1)
    z=bp_logits(warm_syndrome);decode(PK,warm_syndrome,z,Q,'syndrome',8,'prior',False)
    if barrier is not None:barrier.wait(timeout=180)

def bp_logits(s):
    BP.decode(s.astype(np.uint8));z=-np.array(BP.log_prob_ratios,dtype=float,copy=True)
    return np.nan_to_num(np.where(np.isnan(z),np.log(Q/(1-Q)),z),posinf=1e6,neginf=-1e6)

def task(arg):
    method,s,z=arg
    if method=='BP':z=bp_logits(s)
    return decode(PK,s,z,Q,'syndrome',8,'prior',False)[0]

def stats(seconds,frames=1):
    a=np.asarray(seconds,dtype=float)
    return dict(mean_ms=float(a.mean()*1000),median_ms=float(np.median(a)*1000),p95_ms=float(np.quantile(a,.95)*1000),measurements=len(a),frames_per_call=frames,amortized_ms_per_frame=float(a.mean()*1000/frames),frames_per_second=float(frames/a.mean()))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:return winreg.QueryValueEx(key,'ProcessorNameString')[0]
    except (ImportError,OSError):return platform.processor()

def main():
    import torch
    from importlib.metadata import version
    from qec_mpt.data import ROOT,CODES,matrices
    from qec_mpt.model import load_model
    from qec_mpt.utils import gpu_features
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/speed')
    p.add_argument('--codes',nargs='+',default=list(CODES),choices=CODES)
    p.add_argument('--workers',nargs='+',type=int,default=[4,8,16,32])
    p.add_argument('--batches',nargs='+',type=int,default=[8,32,64])
    p.add_argument('--batch-shots',type=int,default=512)
    p.add_argument('--single-shots',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=22101116)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert all(args.batch_shots%b==0 for b in args.batches)
    assert max(args.workers)<=os.cpu_count()
    torch.set_num_threads(1)
    protocol=dict(seed=args.seed,p=.005,rounds=12,codes=args.codes,batch_sizes=args.batches,workers=args.workers,batch_shots=args.batch_shots,single_shots=args.single_shots,repeats=args.repeats,hardware=dict(cpu=cpu_name(),logical_CPUs=os.cpu_count(),GPU=torch.cuda.get_device_name(0)),versions={v:version(v) for v in ['torch','ldpc','numpy','scipy','numba']},MPT_precision='BF16 autocast; FP32 conditioning and sparse aggregation',BP_precision='Native LDPC double precision sum-product, independent per-block early stop, max200 parallel schedule, one thread per worker',postprocess='Identical signed zero-centered prior-scored CS8, always executed, shortcutsFalse',clock='Warm synchronous batch service time, host inputs to all host recovery vectors. Includes feature construction, H2D/D2H, process task/result IPC and CS8. Model/graph construction, worker startup, JIT compilation, warmup, evaluation checks, and waiting to accumulate batch excluded.',paired_execution='Shared same-size CPU pool; order of methods randomized per batch. No overlapping batches or hidden GPU/CPU pipelines.',checkpoint_hashes={c:sha(ROOT/'checkpoints'/f'{c}.pt') for c in args.codes})
    pp=args.output/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    for code in args.codes:
        dest=args.output/f'{code}.json'
        if dest.exists():continue
        dat=matrices(code,.005);h,l,q=dat['h'],dat['l'],dat['q'];pk=pack(h)
        a=np.load(ROOT/'data/report_test'/code/'test_0.0050.npz');stored=np.load(ROOT/'results/sum_product'/code/'predictions_0.0050.npz')
        rng=np.random.default_rng(args.seed+int(code[2:]));ix=rng.choice(len(a['labels']),args.batch_shots,replace=False)
        d=a['detectors'][ix];s=d[:,dat['rows']];labels=a['labels'][ix]
        ref={arm:stored[arm+'_prediction'][ix] for arm in ['MPT_CS8','SPA_CS8']}
        model=load_model(code,'cuda',ROOT/'checkpoints'/f'{code}.pt');prior=torch.tensor(np.log(q/(1-q)),device='cuda',dtype=torch.float32)
        initialize(h,q,pk,s[0])
        @torch.inference_mode()
        def neural(indices):
            x=gpu_features(torch.tensor(d[indices],device='cuda'),12,model.n)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x,prior,.005)
            return z.float().cpu().numpy()
        for b in [1,*args.batches]:
            for _ in range(3):neural(np.arange(b))
        single={'BP':[],'MPT':[]};front={'BP':[],'MPT':[]};pred_single={'BP':[],'MPT':[]};idx_single=[]
        for repeat in range(args.repeats):
            for i in rng.permutation(args.single_shots):
                idx_single.append(int(i))
                for method in rng.permutation(['BP','MPT']):
                    torch.cuda.synchronize();start=time.perf_counter()
                    z=bp_logits(s[i]) if method=='BP' else neural([i])[0]
                    middle=time.perf_counter();f=decode(pk,s[i],z,q,'syndrome',8,'prior',False)[0];end=time.perf_counter()
                    single[method].append(end-start);front[method].append(middle-start)
                    assert np.array_equal(h@f%2,s[i]);pred_single[method].append(l@f%2)
            print('SINGLE',code,repeat+1,flush=True)
        result=dict(code=code,p=.005,selected_dataset_indices=ix.tolist(),single={},batch=[])
        for method,arm in [('BP','SPA_CS8'),('MPT','MPT_CS8')]:
            pred=np.asarray(pred_single[method]);ridx=np.asarray(idx_single)
            result['single'][method]=dict(total=stats(single[method]),frontend=stats(front[method]),CS8=stats(np.asarray(single[method])-front[method]),reference_logical_prediction_agreement=float(np.mean(np.all(pred==ref[arm][ridx],axis=1))),sample_FER=float(np.mean(np.any(pred!=labels[ridx],axis=1))))
            if method=='BP':assert np.array_equal(pred,ref[arm][ridx])
        for workers in args.workers:
            barrier=mp.get_context('spawn').Barrier(workers)
            with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(h,q,pk,s[0],barrier)) as pool:
                list(pool.map(task,[('BP',s[0],None)]*(workers*2),chunksize=1))
                # Both message paths warm up before clocks start.
                for b in args.batches:
                    warm_ix=np.arange(b);z=neural(warm_ix)
                    list(pool.map(task,[('BP',s[i],None) for i in warm_ix],chunksize=1))
                    list(pool.map(task,[('MPT',s[i],zz) for i,zz in zip(warm_ix,z)],chunksize=1))
                    values={m:[] for m in ['BP','MPT']};fronts=[];posts=[];predictions={m:[] for m in values};indices=[]
                    for repeat in range(args.repeats):
                        for start_idx in rng.permutation(np.arange(0,args.batch_shots,b)):
                            bix=np.arange(start_idx,start_idx+b);indices.extend(bix.tolist())
                            for method in rng.permutation(['BP','MPT']):
                                torch.cuda.synchronize();begin=time.perf_counter()
                                if method=='MPT':
                                    z=neural(bix);middle=time.perf_counter();jobs=[(method,s[i],zz) for i,zz in zip(bix,z)]
                                else:jobs=[(method,s[i],None) for i in bix]
                                fs=np.asarray(list(pool.map(task,jobs,chunksize=1)));end=time.perf_counter()
                                values[method].append(end-begin)
                                if method=='MPT':fronts.append(middle-begin);posts.append(end-middle)
                                # Correctness checks are outside clocks.
                                assert np.array_equal((h@fs.T).T%2,s[bix])
                                predictions[method].extend((l@fs.T).T%2)
                    row=dict(workers=workers,batch_size=b,methods={})
                    for method,arm in [('BP','SPA_CS8'),('MPT','MPT_CS8')]:
                        pred=np.asarray(predictions[method]);ridx=np.asarray(indices)
                        row['methods'][method]=dict(total=stats(values[method],b),reference_logical_prediction_agreement=float(np.mean(np.all(pred==ref[arm][ridx],axis=1))),sample_FER=float(np.mean(np.any(pred!=labels[ridx],axis=1))))
                        if method=='BP':assert np.array_equal(pred,ref[arm][ridx])
                    row['methods']['MPT'].update(frontend=stats(fronts,b),parallel_CS8_and_IPC=stats(posts,b))
                    row['BP_over_MPT_speed_ratio']=row['methods']['BP']['total']['amortized_ms_per_frame']/row['methods']['MPT']['total']['amortized_ms_per_frame']
                    result['batch'].append(row)
                    np.savez_compressed(args.output/f'{code}_w{workers}_b{b}.npz',BP_seconds=values['BP'],MPT_seconds=values['MPT'],MPT_frontend_seconds=fronts,MPT_post_seconds=posts,dataset_indices=ix[np.asarray(indices)],BP_logical_predictions=np.asarray(predictions['BP']),MPT_logical_predictions=np.asarray(predictions['MPT']))
                    (args.output/f'{code}_progress.json').write_text(json.dumps(result,indent=2))
                    print('BATCH',code,workers,b,'BP_ms',row['methods']['BP']['total']['amortized_ms_per_frame'],'MPT_ms',row['methods']['MPT']['total']['amortized_ms_per_frame'],'ratio',row['BP_over_MPT_speed_ratio'],flush=True)
            # Pool is closed before the next resource setting.
        np.savez_compressed(args.output/f'{code}_single.npz',BP_seconds=single['BP'],MPT_seconds=single['MPT'],dataset_indices=ix[np.asarray(idx_single)],BP_logical_predictions=np.asarray(pred_single['BP']),MPT_logical_predictions=np.asarray(pred_single['MPT']))
        dest.write_text(json.dumps(result,indent=2));del model;torch.cuda.empty_cache()
        print('CODE_COMPLETE',code,flush=True)
    results=[json.loads((args.output/f'{c}.json').read_text()) for c in args.codes]
    (args.output/'results.json').write_text(json.dumps(dict(protocol=protocol,results=results),indent=2))
    print('SPEED_BENCHMARK_COMPLETE',flush=True)

if __name__=='__main__':main()
