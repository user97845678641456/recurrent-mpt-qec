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

def official_core(H,width=64,depth=2):
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
    return upstream.ECC_Transformer(SimpleNamespace(code=code,d_model=width,N_dec=depth,h=4))

class ResidualScale(nn.Module):
    def __init__(self, dropout, scale):
        super().__init__()
        self.dropout, self.scale = dropout, scale

    def forward(self, x):
        return self.dropout(x) * self.scale


class RecurrentMPT(nn.Module):
    def __init__(self,dat,width=64,depth=2,fault_neighbors=False,residual_scale=1.):
        super().__init__();b=dat['b'];self.n=b.Hx.shape[1];self.d=width;self.rounds=dat['meta']['rounds']
        self.node=nn.Parameter(torch.randn(1,2*self.n,width)*.02)
        self.embed=nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,width))
        core=official_core(np.vstack((b.Hz,b.Hx)),width,depth);self.encoder=core.decoder
        if residual_scale != 1.:
            for layer in self.encoder.layers:
                for sublayer in layer.sublayer:
                    sublayer.dropout = ResidualScale(sublayer.dropout, residual_scale)
        self.register_buffer('mask_vn',core.src_mask_VN);self.register_buffer('mask_cn',core.src_mask_CN)
        h=dat['h'].tocoo();extended=sp.coo_matrix((h.data,(dat['rows'][h.row],h.col)),shape=((self.rounds+1)*self.n,h.shape[1])).tocsc()
        a=extended.T.astype(np.float32).tocsr();degree=np.asarray(a.sum(1)).ravel()
        self.register_buffer('incidence',sparse_tensor(sp.diags(1/degree.clip(1))@a))
        self.register_buffer('raw_incidence',sparse_tensor(a))
        self.register_buffer('prior',torch.tensor(np.log(dat['q']/(1-dat['q'])),dtype=torch.float32))
        self.fault_id=nn.Parameter(torch.randn(len(dat['q']),16)*.02)
        self.local=nn.Linear(width,16);self.global_proj=nn.Linear(2*width,16)
        self.fault_head=nn.Sequential(nn.Linear(50,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.fault_head[-1].weight);nn.init.zeros_(self.fault_head[-1].bias)
        self.noise_condition=nn.Linear(1,2*width,bias=False);nn.init.zeros_(self.noise_condition.weight)

        if fault_neighbors:
            typed = fault_neighbors == 'typed'
            if typed:
                # The author circuit orders its two detector types in contiguous halves.
                kinds = np.arange(self.n) >= self.n // 2
                same = kinds[:, None] == kinds[None, :]
            h=dat['hf'].astype(np.float32).tocsr()
            blocks=[]
            for t in range(self.rounds+1):
                ht=h[t*self.n:(t+1)*self.n]
                a=(ht @ ht.T).toarray()
                np.fill_diagonal(a,0)
                a/=a.sum(1,keepdims=True).clip(1)
                blocks.append(np.stack((a * same, a * ~same)) if typed else a)
            self.register_buffer('fault_neighbors',torch.tensor(np.stack(blocks)),persistent=False)
            self.fault_mix=nn.Linear(width,width,bias=False)
            nn.init.zeros_(self.fault_mix.weight)
            if typed:
                self.fault_mix_cross=nn.Linear(width,width,bias=False)
                nn.init.zeros_(self.fault_mix_cross.weight)

    def clear_cache(self):
        for layer in self.encoder.layers:layer.self_attn.attn=None

    def forward(self,x,prior,noise_p):
        bs=len(x);v=torch.zeros(bs,self.n,self.d,device=x.device);c=torch.zeros_like(v);frames=[]
        p=torch.full((bs,1),noise_p,device=x.device,dtype=torch.float32)
        with torch.autocast('cuda',enabled=False):cond=self.noise_condition(torch.logit(p)-math.log(.003/(1-.003)))
        ev=cond[:,:self.d,None].transpose(1,2);ec=cond[:,self.d:,None].transpose(1,2)
        def encode(v,c):
            result=self.encoder(v,c,self.mask_vn,self.mask_cn);self.clear_cache();return result
        for t in range(self.rounds+1):
            feat=self.embed(x[:,t])
            if hasattr(self,'fault_mix_cross'):
                same, cross = self.fault_neighbors[t].unbind(0)
                feat=feat+self.fault_mix(same @ feat)+self.fault_mix_cross(cross @ feat)
            elif hasattr(self,'fault_mix'):
                feat=feat+self.fault_mix(self.fault_neighbors[t] @ feat)
            v=(v+self.node[:,:self.n]+ev)/math.sqrt(2);c=(c+feat+self.node[:,self.n:]+ec)/math.sqrt(2)
            v,c=checkpoint(encode,v,c,use_reentrant=False) if self.training else encode(v,c);frames.append(c)
        local=self.local(torch.cat(frames,1))
        with torch.autocast('cuda',enabled=False):
            loc=torch.sparse.mm(self.incidence,local.float().permute(1,0,2).reshape(-1,bs*16)).reshape(len(self.prior),bs,16).permute(1,0,2)
            active=torch.sparse.mm(self.raw_incidence,x[...,1].flatten(1,2).float().T).T.unsqueeze(-1)
        glob=self.global_proj(torch.cat((v.mean(1),c.mean(1)),-1))[:,None].expand(-1,len(self.prior),-1)
        feat=torch.cat((loc,glob,self.fault_id[None].expand(bs,-1,-1),active,prior[None,:,None].expand(bs,-1,-1)/10),-1)
        return prior+self.fault_head(feat).squeeze(-1)


def fault_parameters(parameters):
    """Drop only known historical auxiliary-head keys; strict loading follows."""
    return {k:v for k,v in parameters.items() if k not in ('support','query') and not k.startswith(('read.','qnorm.','head.'))}

FINAL_DEPTH = {"bb72_r6": 4, "bb144": 6}


def load_model(code, device="cpu", checkpoint_path=None):
    model = RecurrentMPT(matrices(code), width=96, depth=FINAL_DEPTH[code], fault_neighbors=True)
    state = torch.load(checkpoint_path or ROOT / "checkpoints" / f"{code}.pt",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state["parameters"])
    return model.to(device).eval()


@torch.inference_mode()
def infer(code,p,detectors,device='cpu',checkpoint_path=None,batch=64):
    model=load_model(code,device,checkpoint_path);q=matrices(code,p)['q'];prior=torch.tensor(np.log(q/(1-q)),device=device,dtype=torch.float32);out=[]
    for i in range(0,len(detectors),batch):
        x=gpu_features(torch.tensor(detectors[i:i+batch],device=device),model.rounds,model.n)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):z=model(x,prior,p)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)
