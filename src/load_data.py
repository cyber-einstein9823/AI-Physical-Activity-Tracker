from pathlib import Path
import numpy as np

CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]

def find_root(data_dir="data"):
    base=Path(data_dir)
    for d in sorted(base.rglob("Inertial Signals")):
        if d.parent.name == "train": return d.parent.parent
    raise FileNotFoundError(f"Could not find train/Inertial Signals under {base}. Extract UCI HAR Dataset.zip first.")

def load_raw(data_dir="data"):
    root=find_root(data_dir); out={}
    for split in ("train","test"):
        sig=root/split/"Inertial Signals"
        X=np.stack([np.loadtxt(sig/f"{c}_{split}.txt",dtype=np.float32) for c in CHANNELS],axis=-1)
        y=np.loadtxt(root/split/f"y_{split}.txt",dtype=np.int64)
        subject=np.loadtxt(root/split/f"subject_{split}.txt",dtype=np.int64)
        out[split]={"X":X,"y":y,"subject":subject}
    out["activity_names"]=[line.split()[1] for line in (root/"activity_labels.txt").read_text().splitlines() if line.strip()]
    return out

if __name__=="__main__":
    raw=load_raw()
    for s in ("train","test"):
        d=raw[s]; print(f"{s}: X {d['X'].shape}, y {d['y'].shape}, subject {d['subject'].shape}; subjects={sorted(set(d['subject'].tolist()))}")
    print("Activities:",raw["activity_names"])
