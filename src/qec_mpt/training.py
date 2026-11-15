"""Online ordinary-BCE training, independent validation, resumable state."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from .data import matrices,POINTS
from .model import RecurrentMPT
from .utils import sparse_tensor,gpu_features

def training_data(code,device):
    base=matrices(code);ok=base['mapping']>=0
    mt=sparse_tensor(sp.coo_matrix((np.ones(ok.sum()),(base['mapping'][ok],np.flatnonzero(ok))),shape=(len(base['q']),len(base['pf'])))).to(device)
    ht=sparse_tensor(base['hf']).to(device);lt=sparse_tensor(base['lf']).to(device)
    probs={};priors={};expected={}
    for p in POINTS:
        dat=matrices(code,p)
        assert (dat['hf']!=base['hf']).nnz==0 and (dat['h']!=base['h']).nnz==0 and (dat['lf']!=base['lf']).nnz==0 and (dat['l']!=base['l']).nnz==0
        assert np.array_equal(dat['mapping'],base['mapping'])
        probs[p]=torch.tensor(dat['pf'],device=device,dtype=torch.float32)
        priors[p]=torch.tensor(np.log(dat['q']/(1-dat['q'])),device=device,dtype=torch.float32);expected[p]=float(dat['q'].sum())
    @torch.no_grad()
    def sample(bs,p,gen):
        f=(torch.rand(bs,len(base['pf']),device=device,generator=gen)<probs[p]).float()
        return torch.sparse.mm(ht,f.T).T.remainder(2),torch.sparse.mm(lt,f.T).T.remainder(2),torch.sparse.mm(mt,f.T).T.remainder(2)
    return sample,priors,expected

def train(args):
    device=args.device;folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);model=RecurrentMPT(matrices(args.code)).to(device)
    if args.initialize:model.load_state_dict(torch.load(args.initialize,map_location=device,weights_only=True)['parameters'])
    ema=copy.deepcopy(model).eval();sample,priors,expected=training_data(args.code,device)
    gen=torch.Generator(device=device).manual_seed(args.seed+5000);chooser=np.random.default_rng(args.seed+6000)
    vg=torch.Generator(device=device).manual_seed(args.seed+4000)
    validation={p:sample(args.validation_shots,p,vg) for p in POINTS}
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    history=[];seen=0;best=float('inf');last=folder/'last.pt'
    if last.exists():
        st=torch.load(last,map_location=device,weights_only=False)
        assert st['config']==vars(args),'Resume arguments differ from saved run'
        model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer']);seen=st['samples'];best=st['best'];history=st['history'];gen.set_state(st['generator'].cpu());chooser.bit_generator.state=st['chooser'];torch.set_rng_state(st['cpu_rng'].cpu())
        if str(device).startswith('cuda'):torch.cuda.set_rng_state(st['cuda_rng'].cpu())
    amp=str(device).startswith('cuda')
    def weights(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    @torch.no_grad()
    def validate():
        ema.eval();values={}
        for p,(d,y,g) in validation.items():
            total=0.
            for i in range(0,len(d),64):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z,_=ema(gpu_features(d[i:i+64],12,model.n),priors[p],p)
                total+=float(F.binary_cross_entropy_with_logits(z.float(),g[i:i+64],reduction='sum'))
            values[str(p)]=total/len(d)/expected[p]
        return values,float(np.mean(list(values.values())))
    if seen==0:
        v,best=validate();torch.save(dict(parameters=weights(ema),samples=0,validation=v),folder/'best.pt')
    begin=time.perf_counter();nextcheck=(seen//args.eval_every+1)*args.eval_every
    while seen<args.samples:
        bs=min(args.batch,args.samples-seen);p=float(chooser.choice(POINTS));d,y,g=sample(bs,p,gen);model.train();opt.zero_grad(set_to_none=True)
        lr=args.lr*(.1+.9*.5*(1+math.cos(math.pi*seen/args.samples))) if args.schedule=='cosine' else max(args.lr/10,args.lr*.5**(seen//200000))
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z,logical=model(gpu_features(d,12,model.n),priors[p],p)
        loss=F.binary_cross_entropy_with_logits(z.float(),g,reduction='sum')/bs/expected[p]+.2*F.binary_cross_entropy_with_logits(logical.float(),y)
        assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.01)
        seen+=bs
        if seen>=nextcheck or seen==args.samples:
            v,score=validate();history.append(dict(samples=seen,validation=v,score=score,lr=lr,seconds=time.perf_counter()-begin));print(json.dumps(history[-1]),flush=True)
            if score<best:
                best=score;torch.save(dict(parameters=weights(ema),samples=seen,validation=v,selection='minimum independent validation mean BCE'),folder/'best.pt')
            st=dict(model=weights(model),ema=weights(ema),optimizer=opt.state_dict(),samples=seen,best=best,history=history,generator=gen.get_state(),chooser=chooser.bit_generator.state,cpu_rng=torch.get_rng_state(),config=vars(args))
            if amp:st['cuda_rng']=torch.cuda.get_rng_state()
            torch.save(st,last);(folder/'history.json').write_text(json.dumps(history,indent=2));nextcheck+=args.eval_every
    torch.save(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),folder/'final.pt')
