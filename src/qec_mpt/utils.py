"""DEM construction and feature transforms used in the report."""
import numpy as np
import scipy.sparse as sp
import stim
import torch

def dem_matrices(circuit):
    dem=circuit.detector_error_model().flattened();cols={}
    for op in dem:
        if op.type!='error':continue
        ds=set();ls=set()
        for t in op.targets_copy():
            if t.is_relative_detector_id():
                if t.val in ds:ds.remove(t.val)
                else:ds.add(t.val)
            elif t.is_logical_observable_id():
                if t.val in ls:ls.remove(t.val)
                else:ls.add(t.val)
        if not ds and not ls:continue
        key=(tuple(sorted(ds)),tuple(sorted(ls)));p=op.args_copy()[0]
        cols[key]=cols.get(key,0)*(1-p)+(1-cols.get(key,0))*p
    rr=[];cc=[];lr=[];lc=[];ps=[]
    for j,((ds,ls),p) in enumerate(cols.items()):
        rr.extend(ds);cc.extend([j]*len(ds));lr.extend(ls);lc.extend([j]*len(ls));ps.append(p)
    h=sp.csc_matrix((np.ones(len(rr),np.uint8),(rr,cc)),shape=(circuit.num_detectors,len(ps)))
    l=sp.csc_matrix((np.ones(len(lr),np.uint8),(lr,lc)),shape=(circuit.num_observables,len(ps)))
    return h,l,np.array(ps)

def x_circuit(c,n):
    out=stim.Circuit();i=0
    for op in c:
        if op.name=='DETECTOR':
            if i%n>=n//2:out.append(op)
            i+=1
        else:out.append(op)
    return out

def keys(h,l,offset=0):
    h=h.tocsc();l=l.tocsc()
    return [(tuple((h.indices[h.indptr[j]:h.indptr[j+1]]+offset).tolist()),tuple(l.indices[l.indptr[j]:l.indptr[j+1]].tolist())) for j in range(h.shape[1])]

def gpu_features(det,rounds,n):
    ev=det.reshape(-1,rounds+1,n).float();valid=torch.ones_like(ev);final=torch.zeros_like(ev);final[:,-1]=1
    return torch.stack((ev.cumsum(1)%2,ev,valid,final),-1)

def sparse_tensor(a):
    torch.sparse.check_sparse_tensor_invariants.enable()
    a=a.tocoo();idx=torch.tensor(np.vstack((a.row,a.col)),dtype=torch.int64)
    return torch.sparse_coo_tensor(idx,torch.tensor(a.data,dtype=torch.float32),a.shape,check_invariants=True).coalesce()
