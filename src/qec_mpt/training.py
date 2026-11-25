"""Online ordinary-BCE training, independent validation, resumable state."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from .data import matrices,POINTS
from .model import RecurrentMPT,fault_parameters
from .utils import sparse_tensor,gpu_features

def training_data(code,device):
    base=matrices(code);ok=base['mapping']>=0
    mt=sparse_tensor(sp.coo_matrix((np.ones(ok.sum()),(base['mapping'][ok],np.flatnonzero(ok))),shape=(len(base['q']),len(base['pf'])))).to(device)
    ht=sparse_tensor(base['hf']).to(device)
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
        return torch.sparse.mm(ht,f.T).T.remainder(2),torch.sparse.mm(mt,f.T).T.remainder(2)
    return sample,priors,expected

def train(args):
    rank_weight=getattr(args,"rank_weight",0.)
    objective="fault_BCE_plus_rank" if rank_weight else "fault_BCE_only"
    device=args.device;folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);model=RecurrentMPT(matrices(args.code),width=getattr(args,"width",64),depth=getattr(args,"depth",2),fault_neighbors=getattr(args,"fault_neighbors",False),residual_scale=getattr(args,"residual_scale",1.)).to(device)
    if args.initialize:model.load_state_dict(fault_parameters(torch.load(args.initialize,map_location=device,weights_only=True)['parameters']))
    ema=copy.deepcopy(model).eval();sample,priors,expected=training_data(args.code,device)
    gen=torch.Generator(device=device).manual_seed(args.seed+5000);chooser=np.random.default_rng(args.seed+6000)
    vg=torch.Generator(device=device).manual_seed(args.seed+4000)
    validation={p:sample(args.validation_shots,p,vg) for p in POINTS}
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    history=[];seen=0;best=float('inf');optimizer_stage=0;elapsed=0.;last=folder/'last.pt'
    if last.exists():
        st=torch.load(last,map_location=device,weights_only=False)
        assert st.get('training_loss')==objective,'Cannot resume a historical auxiliary-loss optimizer; start a new run with --initialize'
        assert st['config']==vars(args),'Resume arguments differ from saved run'
        model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer']);seen=st['samples'];best=st['best'];history=st['history'];gen.set_state(st['generator'].cpu());chooser.bit_generator.state=st['chooser'];torch.set_rng_state(st['cpu_rng'].cpu())
        optimizer_stage=st.get('optimizer_stage',0);elapsed=st.get('elapsed',0.)
        if str(device).startswith('cuda'):torch.cuda.set_rng_state(st['cuda_rng'].cpu())
    amp=str(device).startswith('cuda')
    def weights(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    @torch.no_grad()
    def validate():
        ema.eval();values={}
        for p,(d,g) in validation.items():
            total=0.
            for i in range(0,len(d),64):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z=ema(gpu_features(d[i:i+64],model.rounds,model.n),priors[p],p)
                total+=float(F.binary_cross_entropy_with_logits(z.float(),g[i:i+64],reduction='sum'))
            values[str(p)]=total/len(d)/expected[p]
        return values,float(np.mean(list(values.values())))
    if seen==0:
        v,best=validate();torch.save(dict(parameters=weights(ema),samples=0,validation=v,training_loss=objective,initialization=args.initialize),folder/'best.pt')
    begin=time.perf_counter();nextcheck=(seen//args.eval_every+1)*args.eval_every
    stopping_reason='sample_cap'
    grad_every=getattr(args,'grad_every',0)
    step=0;grad_window=[]
    while seen<args.samples:
        bs=min(args.batch,args.samples-seen);p=float(chooser.choice(POINTS));d,g=sample(bs,p,gen);model.train();opt.zero_grad(set_to_none=True)
        if args.schedule=='two_stage':
            if seen>=100000 and optimizer_stage==0:
                opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);optimizer_stage=1
            lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*seen/100000))) if seen<100000 else max(1e-5,1e-4*.5**((seen-100000)//200000))
        else:lr=args.lr*(.1+.9*.5*(1+math.cos(math.pi*seen/args.samples))) if args.schedule=='cosine' else max(args.lr/10,args.lr*.5**(seen//200000))
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):z=model(gpu_features(d,model.rounds,model.n),priors[p],p)
        loss=F.binary_cross_entropy_with_logits(z.float(),g,reduction='sum')/bs/expected[p]
        if rank_weight:
            logits=z.float()
            conf=logits.abs()
            wrong=(logits.detach()>0)!=g.bool()
            # Compare the most confident mistakes with near-boundary correct predictions.
            bad=conf.masked_fill(~wrong,-torch.inf).topk(16,dim=1).values
            good=conf.masked_fill(wrong,torch.inf).topk(16,dim=1,largest=False).values
            rank_loss=F.relu(0.5+bad-good).mean()
            loss=loss+rank_weight*rank_loss
        assert torch.isfinite(loss);loss.backward()
        step+=1
        monitor=grad_every and (step==1 or step%grad_every==0)
        if monitor:
            layer_rms=[]
            for layer in model.encoder.layers:
                grads=[p.grad.detach() for p in layer.parameters() if p.grad is not None]
                rms=(torch.stack([g.float().norm().square() for g in grads]).sum()
                     /sum(g.numel() for g in grads)).sqrt()
                layer_rms.append(rms)
        norm=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm)
        if monitor:
            row=dict(samples=seen+bs, rms=torch.stack(layer_rms).cpu().tolist(),
                     global_norm=norm.item())
            grad_window.append(row)
            with (folder/'gradients.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
        opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.01)
        seen+=bs
        if seen>=nextcheck or seen==args.samples:
            v,score=validate();seconds=elapsed+time.perf_counter()-begin;history.append(dict(samples=seen,validation=v,score=score,lr=lr,seconds=seconds))
            if grad_window:
                history[-1]['gradients']=dict(
                    observations=len(grad_window),
                    layer_rms_median=np.median([r['rms'] for r in grad_window],axis=0).tolist(),
                    sampled_clip_fraction=float(np.mean([r['global_norm']>1 for r in grad_window])))
                grad_window.clear()
            print(json.dumps(dict(code=args.code,**history[-1])),flush=True)
            if score<best:
                best=score;torch.save(dict(parameters=weights(ema),samples=seen,validation=v,selection='minimum independent validation mean BCE',training_loss=objective,initialization=args.initialize),folder/'best.pt')
            st=dict(model=weights(model),ema=weights(ema),optimizer=opt.state_dict(),optimizer_stage=optimizer_stage,elapsed=seconds,samples=seen,best=best,history=history,generator=gen.get_state(),chooser=chooser.bit_generator.state,cpu_rng=torch.get_rng_state(),config=vars(args),training_loss=objective)
            if amp:st['cuda_rng']=torch.cuda.get_rng_state()
            torch.save(st,last);(folder/'history.json').write_text(json.dumps(history,indent=2));nextcheck+=args.eval_every
            patience=getattr(args,'plateau_patience',0)
            if patience and seen>=getattr(args,'plateau_min_samples',0) and len(history)>patience:
                previous=min(r['score'] for r in history[:-patience])
                recent=min(r['score'] for r in history[-patience:])
                if (previous-recent)/previous<getattr(args,'plateau_relative',.002):
                    stopping_reason='validation_practical_plateau'
                    break
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    selected.update(code=args.code,seed=args.seed,training_samples_executed=seen,schedule=args.schedule,stopping_reason=stopping_reason)
    torch.save(selected,folder/'final.pt')
