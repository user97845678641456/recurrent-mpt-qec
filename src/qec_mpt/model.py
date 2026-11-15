"""Report's ungated two-layer recurrent model; upstream attention fetched separately."""
import importlib.util,json,sys,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .data import ROOT,matrices
from .utils import sparse_tensor,gpu_features

def official_core(H):
    folder=ROOT/'third_party/crossmpt'
    if not (folder/'Model_CrossMPT.py').exists():raise FileNotFoundError('Run python scripts/fetch_upstream.py first. See THIRD_PARTY.md.')
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    codes=module('_qec_upstream_codes',folder/'Codes.py');previous=sys.modules.get('Codes');sys.modules['Codes']=codes
    try:upstream=module('_qec_upstream_crossmpt',folder/'Model_CrossMPT.py')
    finally:
        if previous is None:sys.modules.pop('Codes',None)
        else:sys.modules['Codes']=previous
    code=SimpleNamespace(n=H.shape[1],pc_matrix=torch.tensor(H,dtype=torch.long))
    return upstream.ECC_Transformer(SimpleNamespace(code=code,d_model=64,N_dec=2,h=4))

class RecurrentMPT(nn.Module):
    def __init__(self,dat):
        super().__init__();b=dat['b'];self.n=b.Hx.shape[1];self.d=64;self.rounds=12
        self.register_buffer('support',torch.tensor(b.Lx,dtype=torch.float32))
        self.node=nn.Parameter(torch.randn(1,2*self.n,64)*.02)
        self.embed=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,64))
        self.query=nn.Parameter(torch.randn(1,len(b.Lx),64)*.02)
        self.read=nn.MultiheadAttention(64,4,batch_first=True);self.qnorm=nn.LayerNorm(64)
        self.head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))
        core=official_core(np.vstack((b.Hz,b.Hx)));self.encoder=core.decoder
        self.register_buffer('mask_vn',core.src_mask_VN);self.register_buffer('mask_cn',core.src_mask_CN)
        h=dat['h'].tocoo();extended=sp.coo_matrix((h.data,(dat['rows'][h.row],h.col)),shape=(13*self.n,h.shape[1])).tocsc()
        a=extended.T.astype(np.float32).tocsr();degree=np.asarray(a.sum(1)).ravel()
        self.register_buffer('incidence',sparse_tensor(sp.diags(1/degree.clip(1))@a))
        self.register_buffer('raw_incidence',sparse_tensor(a))
        self.register_buffer('prior',torch.tensor(np.log(dat['q']/(1-dat['q'])),dtype=torch.float32))
        self.fault_id=nn.Parameter(torch.randn(len(dat['q']),16)*.02)
        self.local=nn.Linear(64,16);self.global_proj=nn.Linear(128,16)
        self.fault_head=nn.Sequential(nn.Linear(50,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.fault_head[-1].weight);nn.init.zeros_(self.fault_head[-1].bias)
        self.noise_condition=nn.Linear(1,128,bias=False);nn.init.zeros_(self.noise_condition.weight)

    def clear_cache(self):
        for layer in self.encoder.layers:layer.self_attn.attn=None

    def forward(self,x,prior,noise_p):
        bs=len(x);v=torch.zeros(bs,self.n,self.d,device=x.device);c=torch.zeros_like(v);frames=[]
        p=torch.full((bs,1),noise_p,device=x.device,dtype=torch.float32)
        with torch.autocast('cuda',enabled=False):cond=self.noise_condition(torch.logit(p)-math.log(.003/(1-.003)))
        ev=cond[:,:self.d,None].transpose(1,2);ec=cond[:,self.d:,None].transpose(1,2)
        def encode(v,c):
            result=self.encoder(v,c,self.mask_vn,self.mask_cn);self.clear_cache();return result
        for t in range(13):
            v=(v+self.node[:,:self.n]+ev)/math.sqrt(2);c=(c+self.embed(x[:,t])+self.node[:,self.n:]+ec)/math.sqrt(2)
            v,c=checkpoint(encode,v,c,use_reentrant=False) if self.training else encode(v,c);frames.append(c)
        q=self.query+(self.support@v)/self.support.sum(1)[None,:,None];mem=torch.cat((v,c),1);q=q+self.read(q,mem,mem,need_weights=False)[0]
        logical=self.head(self.qnorm(q)).squeeze(-1);local=self.local(torch.cat(frames,1))
        with torch.autocast('cuda',enabled=False):
            loc=torch.sparse.mm(self.incidence,local.float().permute(1,0,2).reshape(-1,bs*16)).reshape(len(self.prior),bs,16).permute(1,0,2)
            active=torch.sparse.mm(self.raw_incidence,x[...,1].flatten(1,2).float().T).T.unsqueeze(-1)
        glob=self.global_proj(torch.cat((v.mean(1),c.mean(1)),-1))[:,None].expand(-1,len(self.prior),-1)
        feat=torch.cat((loc,glob,self.fault_id[None].expand(bs,-1,-1),active,prior[None,:,None].expand(bs,-1,-1)/10),-1)
        return prior+self.fault_head(feat).squeeze(-1),logical

def load_model(code,device='cpu',checkpoint_path=None):
    model=RecurrentMPT(matrices(code)).to(device)
    state=torch.load(checkpoint_path or ROOT/'checkpoints'/f'{code}.pt',map_location=device,weights_only=True)
    model.load_state_dict(state['parameters']);return model.eval()

@torch.inference_mode()
def infer(code,p,detectors,device='cpu',checkpoint_path=None,batch=64):
    model=load_model(code,device,checkpoint_path);q=matrices(code,p)['q'];prior=torch.tensor(np.log(q/(1-q)),device=device,dtype=torch.float32);out=[]
    for i in range(0,len(detectors),batch):
        x=gpu_features(torch.tensor(detectors[i:i+batch],device=device),12,model.n)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):z,_=model(x,prior,p)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)
