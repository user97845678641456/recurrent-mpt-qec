"""Packed GF(2) OSD for large DEM matrices, with external posterior logits.

OSD0 and CS8 share this backend for MPT and BP. Default scoring uses the
nonuniform DEM prior; optional posterior scoring uses supplied logits. CS8 means all
single free-variable flips and pairs among the first eight free variables.
"""
import os
from pathlib import Path
import numpy as np
from numba import njit
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[2]/'.numba_cache'))


def pack(h):
    a=h.toarray().astype(np.uint8) if hasattr(h,'toarray') else np.asarray(h,dtype=np.uint8)
    rows=np.zeros((a.shape[0],(a.shape[1]+63)//64),np.uint64)
    for j in range(a.shape[1]):rows[:,j//64]|=a[:,j].astype(np.uint64)<<np.uint64(j%64)
    return rows

@njit(cache=True)
def bit(row,j):return (row[j//64]>>np.uint64(j%64))&np.uint64(1)

@njit(cache=True)
def one(packed,syndrome,z,weights,centered,sweep,shortcuts=True):
    m=packed.shape[0];n=len(z);a=packed.copy();rhs=syndrome.copy()
    hard=(z>0).astype(np.uint8) if centered else np.zeros(n,np.uint8)
    if shortcuts and not np.any(syndrome):return np.zeros(n,np.uint8),1
    if centered:
        for j in range(n):
            if hard[j]:
                for r in range(m):rhs[r]^=np.uint8(bit(a[r],j))
        if shortcuts and not np.any(rhs):return hard,1
    order=np.argsort(np.abs(z) if centered else -z,kind='mergesort')
    piv=np.empty(m,np.int64);isp=np.zeros(n,np.uint8);rank=0
    for col in order:
        pr=-1
        for r in range(rank,m):
            if bit(a[r],col):pr=r;break
        if pr<0:continue
        tmp=a[rank].copy();a[rank]=a[pr];a[pr]=tmp
        t=rhs[rank];rhs[rank]=rhs[pr];rhs[pr]=t
        for r in range(m):
            if r!=rank and bit(a[r],col):
                for w in range(a.shape[1]):a[r,w]^=a[rank,w]
                rhs[r]^=rhs[rank]
        piv[rank]=col;isp[col]=1;rank+=1
        if rank==m:break
    for r in range(rank,m):
        if rhs[r]:raise ValueError('Inconsistent syndrome')
    base=hard.copy()
    for r in range(rank):base[piv[r]]^=rhs[r]
    if sweep==0:return base,1
    free=np.empty(n-rank,np.int64);k=0
    for col in order:
        if not isp[col]:free[k]=col;k+=1
    # Cost changes from toggling each variable relative to base.
    cost=weights*(1.-2.*base)
    best=0.;best_a=-1;best_b=-1
    for j in range(k):
        col=free[j];delta=cost[col]
        for r in range(rank):
            if bit(a[r],col):delta+=cost[piv[r]]
        if delta<best-1e-10:best=delta;best_a=col;best_b=-1
    depth=min(sweep,k)
    for j in range(depth):
        for jj in range(j+1,depth):
            ca=free[j];cb=free[jj];delta=cost[ca]+cost[cb]
            for r in range(rank):
                if bit(a[r],ca)!=bit(a[r],cb):delta+=cost[piv[r]]
            if delta<best-1e-10:best=delta;best_a=ca;best_b=cb
    if best_a>=0:
        base[best_a]^=1
        if best_b>=0:base[best_b]^=1
        for r in range(rank):
            v=bit(a[r],best_a)
            if best_b>=0:v^=bit(a[r],best_b)
            base[piv[r]]^=np.uint8(v)
    return base,1+k+depth*(depth-1)//2

def decode(packed,syndrome,z,ps,basis='syndrome',sweep=0,score='prior',shortcuts=None):
    """Decode using native logits z=log(P1/P0).

    basis='syndrome': zero center; decreasing fault probability chooses pivots.
    basis='hard': start at z>0; ascending abs(z) chooses correction pivots.
    score='prior' uses channel priors ps; 'posterior' uses native logits z.
    Disable shortcuts to compare scores over exactly the same candidate list.
    sweep=8 tests every single free flip and pairs among the first eight.
    """
    if basis not in ('syndrome','hard'):
        raise ValueError("basis must be 'syndrome' or 'hard'")
    if score not in ('prior','posterior'):
        raise ValueError("score must be 'prior' or 'posterior'")
    if shortcuts is None:
        shortcuts=score=='prior'
    z=np.asarray(z,dtype=np.float64)
    prior=np.log(ps/(1-ps))
    z=np.where(np.isnan(z),prior,z)
    z=np.nan_to_num(z,posinf=1e6,neginf=-1e6)
    return one(packed,np.asarray(syndrome,dtype=np.uint8),z,-prior if score=='prior' else -z,basis=='hard',sweep,shortcuts)
