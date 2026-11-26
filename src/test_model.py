from pathlib import Path
import sys, json, csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).parent))
from model import HARNet
from metrics import accuracy,confusion_matrix,per_class_accuracy

def main():
    Path('results/figures').mkdir(parents=True,exist_ok=True); d=np.load('results/prepared_data.npz',allow_pickle=True)
    ck=torch.load('results/best_model.pt',map_location='cpu',weights_only=False); model=HARNet(); model.load_state_dict(ck['model_state_dict']); model.eval()
    X=torch.tensor(d['X_test'],dtype=torch.float32); y=d['y_test'].astype(int); names=d['activity_names'].tolist();
    with torch.no_grad(): logits,alpha=model(X); probs=torch.softmax(logits,dim=1).numpy(); pred=logits.argmax(1).numpy(); att=alpha.numpy()
    cm=confusion_matrix(y,pred,6); pc=per_class_accuracy(cm); acc=accuracy(y,pred)
    np.savez_compressed('results/test_predictions.npz',predictions=pred,probabilities=probs,attention=att)
    with open('results/test_metrics.json','w') as f: json.dump({'test_accuracy':acc,'per_class_accuracy':pc.tolist(),'confusion_matrix':cm.tolist(),'best_epoch':int(ck['epoch']),'best_validation_loss':float(ck['val_loss'])},f,indent=2)
    with open('results/per_class_metrics.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['label','activity','accuracy','test_count']); w.writerows([[i+1,n,float(pc[i]),int(cm[i].sum())] for i,n in enumerate(names)])
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cm); ax.set_xticks(range(6),names,rotation=45,ha='right'); ax.set_yticks(range(6),names); ax.set_xlabel('Predicted activity'); ax.set_ylabel('True activity'); ax.set_title('HAR Test Confusion Matrix');
    for i in range(6):
        for j in range(6): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.colorbar(im); fig.tight_layout(); fig.savefig('results/figures/confusion_matrix.png',dpi=160); plt.close(fig)
    hist=np.genfromtxt('results/training_history.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(hist['epoch'],hist['train_loss'],label='Training loss'); ax.plot(hist['epoch'],hist['val_loss'],label='Validation loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig('results/figures/loss_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(hist['epoch'],hist['val_accuracy'],label='Validation accuracy'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig('results/figures/validation_accuracy.png',dpi=160); plt.close(fig)
    # Task 5: one correctly classified example/activity with body acceleration and attention.
    t=np.arange(128)/50; fig,axes=plt.subplots(6,1,figsize=(11,14),sharex=True)
    avg_attention=[]; mean_peak=[]
    for k,name in enumerate(names):
        idxs=np.where(y==k)[0]; correct=idxs[pred[idxs]==k]
        idx=int(correct[0]) if len(correct) else int(idxs[0]); avg=att[idxs].mean(axis=0); avg_attention.append(avg); mean_peak.append(float(att[idxs].max(axis=1).mean()))
        axes[k].plot(t,X[idx,:,0].numpy(),label='Body acc X'); axes[k].plot(t,X[idx,:,1].numpy(),label='Body acc Y'); axes[k].plot(t,X[idx,:,2].numpy(),label='Body acc Z'); axes[k].set_title(f'{name} — prediction: {names[pred[idx]]}'); axes[k].legend(loc='upper right',fontsize=7); axes[k].twinx().plot(t,att[idx],alpha=.75,linestyle='--',label='Attention');
    axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig('results/figures/attention_examples.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));
    for k,name in enumerate(names): ax.plot(t,avg_attention[k],label=name)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean attention weight'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig('results/figures/average_attention.png',dpi=160); plt.close(fig)
    with open('results/attention_summary.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['activity','mean_max_attention']); w.writerows([[n,v] for n,v in zip(names,mean_peak)])
    print('TEST ACCURACY:',acc); print('PER-CLASS ACCURACY:',dict(zip(names,pc.tolist()))); print('CONFUSION MATRIX (rows=true, columns=pred):\n',cm); print('Best epoch:',ck['epoch']); print('Attention figures and metrics saved.')
if __name__=='__main__': main()
