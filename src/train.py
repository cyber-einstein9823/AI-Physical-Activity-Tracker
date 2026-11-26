from pathlib import Path
import csv, time, sys
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
sys.path.insert(0,str(Path(__file__).parent))
from model import HARNet

def load_npz(): return np.load('results/prepared_data.npz',allow_pickle=True)
def run_training(epochs=30,batch_size=64,lr=1e-3):
    d=load_npz(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    Xtr=torch.tensor(d['X_train'],dtype=torch.float32); ytr=torch.tensor(d['y_train'],dtype=torch.long)
    Xv=torch.tensor(d['X_val'],dtype=torch.float32); yv=torch.tensor(d['y_val'],dtype=torch.long)
    loader=DataLoader(TensorDataset(Xtr,ytr),batch_size=batch_size,shuffle=True)
    torch.manual_seed(0); model=HARNet().to(device); params=sum(p.numel() for p in model.parameters()); print('Trainable parameters:',params)
    criterion=torch.nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    with torch.no_grad(): initial=criterion(model(Xtr[:64].to(device))[0],ytr[:64].to(device)).item()
    print('Initial loss:',initial)
    best=float('inf'); history=[]; start=time.time()
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(); logits,_=model(xb); loss=criterion(logits,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            logits,_=model(Xv.to(device)); val_loss=criterion(logits,yv.to(device)).item(); val_acc=(logits.argmax(1).cpu()==yv).float().mean().item()
        train_loss=float(np.mean(losses)); history.append([epoch,train_loss,val_loss,val_acc]); print(f'Epoch {epoch:02d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | val acc {val_acc:.4f}')
        if val_loss<best:
            best=val_loss; torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'val_loss':val_loss,'initial_loss':initial,'parameter_count':params,'activity_names':d['activity_names'].tolist()},'results/best_model.pt')
    elapsed=time.time()-start; Path('results').mkdir(exist_ok=True)
    with open('results/training_history.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['epoch','train_loss','val_loss','val_accuracy']); w.writerows(history)
    print(f'Training time: {elapsed/60:.2f} minutes; best val loss {best:.4f}; checkpoint saved.')
    return history
if __name__=='__main__': run_training()
