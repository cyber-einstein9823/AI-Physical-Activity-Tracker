from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_raw, CHANNELS

def plot_one_per_activity(data, names, out_path="results/figures/example_windows.png"):
    t=np.arange(128)/50; fig,axes=plt.subplots(2,6,figsize=(19,6),sharex=True)
    for k,name in enumerate(names):
        idx=int(np.flatnonzero(data['y']==k+1)[0])
        for c in range(3): axes[0,k].plot(t,data['X'][idx,:,c],label=CHANNELS[c])
        for c in range(3): axes[1,k].plot(t,data['X'][idx,:,3+c],label=CHANNELS[3+c])
        axes[0,k].set_title(name.replace('_',' '),fontsize=8)
    axes[0,0].set_ylabel('Body acceleration (g)'); axes[1,0].set_ylabel('Gyroscope (rad/s)')
    for ax in axes[1]: ax.set_xlabel('Time (s)')
    axes[0,0].legend(fontsize=6); axes[1,0].legend(fontsize=6)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out_path,dpi=150); plt.close(fig)
    print('Saved',out_path)
if __name__=='__main__':
    raw=load_raw(); plot_one_per_activity(raw['train'],raw['activity_names'])
