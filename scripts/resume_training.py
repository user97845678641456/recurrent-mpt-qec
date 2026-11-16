"""Extend a pure-BCE run in a new directory, retaining optimizer/EMA/RNG."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from qec_mpt.training import train

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,required=True,help='New cumulative sample budget')
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    if source==out:raise ValueError('Use a new output directory to preserve the source run')
    state=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    assert state['training_loss']=='fault_BCE_only'
    assert a.samples>state['samples'],'New budget must exceed source sample count'
    config=dict(state['config']);config.update(output=str(out),samples=a.samples,plateau_patience=0)
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'last.pt').exists():
        if (out/'best.pt').exists() or (out/'final.pt').exists():raise ValueError('Output contains checkpoints without resumable state')
        state['config']=config;torch.save(state,out/'last.pt');shutil.copy2(source/'best.pt',out/'best.pt')
        (out/'resume_provenance.json').write_text(json.dumps(dict(source=str(source),source_samples=state['samples'],config=config),indent=2))
    del state;torch.set_num_threads(4);train(SimpleNamespace(**config))

if __name__=='__main__':main()
