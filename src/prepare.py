from pathlib import Path
import numpy as np
import torch
from load_data import load_raw
VAL_SUBJECTS=[3,15,25,30]

def prepare(data_dir='data', out_dir='results'):
    raw=load_raw(data_dir); tr=raw['train']; te=raw['test']
    val_mask=np.isin(tr['subject'],VAL_SUBJECTS); train_mask=~val_mask
    Xtr, ytr, subtr=tr['X'][train_mask],tr['y'][train_mask]-1,tr['subject'][train_mask]
    Xv, yv, subv=tr['X'][val_mask],tr['y'][val_mask]-1,tr['subject'][val_mask]
    Xte, yte=te['X'],te['y']-1
    mean=Xtr.mean(axis=(0,1),keepdims=True); std=Xtr.std(axis=(0,1),keepdims=True); std=np.where(std<1e-8,1,std)
    Xtr=(Xtr-mean)/std; Xv=(Xv-mean)/std; Xte=(Xte-mean)/std
    # Required shapes and types
    arrays={"X_train":torch.tensor(Xtr,dtype=torch.float32),"y_train":torch.tensor(ytr,dtype=torch.long),"X_val":torch.tensor(Xv,dtype=torch.float32),"y_val":torch.tensor(yv,dtype=torch.long),"X_test":torch.tensor(Xte,dtype=torch.float32),"y_test":torch.tensor(yte,dtype=torch.long),"train_subjects":subtr,"val_subjects":subv,"mean":mean.squeeze(0),"std":std.squeeze(0),"activity_names":raw['activity_names']}
    Path(out_dir).mkdir(parents=True,exist_ok=True); np.savez_compressed(Path(out_dir)/'prepared_data.npz',**{k:(v.numpy() if torch.is_tensor(v) else v) for k,v in arrays.items()})
    print('Raw train:',tr['X'].shape,tr['y'].shape,tr['subject'].shape); print('Raw test:',te['X'].shape,te['y'].shape,te['subject'].shape)
    print(f"Train windows: {len(ytr)} | Validation windows: {len(yv)} | Test windows: {len(yte)}")
    print('Train subjects:',sorted(set(subtr.tolist()))); print('Validation subjects:',sorted(set(subv.tolist()))); print('Test subjects:',sorted(set(te['subject'].tolist())))
    for i,n in enumerate(raw['activity_names']): print(f"{i+1}: {n} | train={int((ytr==i).sum())} val={int((yv==i).sum())} test={int((yte==i).sum())}")
    print('Standardization stats saved (training subset only).')
    return arrays
if __name__=='__main__': prepare()
