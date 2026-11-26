from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model import HARNet
ROOT=Path(__file__).resolve().parents[1]; names=['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
@st.cache_resource
def load_model():
    d=np.load(ROOT/'results/prepared_data.npz',allow_pickle=True); ck=torch.load(ROOT/'results/best_model.pt',map_location='cpu',weights_only=False); model=HARNet(); model.load_state_dict(ck['model_state_dict']); model.eval(); return model,d
st.set_page_config(page_title='HAR Activity Classifier',layout='wide'); st.title('Human Activity Recognition'); st.caption('Educational demonstration using smartphone inertial signals — not for safety or medical decisions.')
model,d=load_model(); option=st.radio('Input mode',['Select a test window','Upload CSV'],horizontal=True)
if option=='Select a test window':
    idx=st.slider('Test window index',0,len(d['X_test'])-1,0); X=d['X_test'][idx]
else:
    up=st.file_uploader('Upload CSV with 128 rows × 9 columns (no header recommended)',type=['csv'])
    if up is None: st.info('Upload a CSV to continue.'); st.stop()
    X=np.loadtxt(up,delimiter=',')
    if X.shape!=(128,9): st.error(f'Expected (128,9), received {X.shape}.'); st.stop()
    # Standardize custom data using training-only stats.
    X=(X-d['mean'])/d['std']
with torch.no_grad(): logits,alpha=model(torch.tensor(X,dtype=torch.float32).unsqueeze(0)); probs=torch.softmax(logits,dim=1).numpy()[0]; pred=int(probs.argmax()); att=alpha.numpy()[0]
col1,col2=st.columns([1,1]);
with col1:
    st.subheader('Predicted activity'); st.success(names[pred]); st.bar_chart(pd.DataFrame({'Probability':probs},index=names));
with col2:
    st.subheader('Signal & attention'); fig,ax=plt.subplots(figsize=(9,4)); t=np.arange(128)/50; ax.plot(t,X[:,0],label='Body Acc X'); ax.plot(t,X[:,1],label='Body Acc Y'); ax.plot(t,X[:,2],label='Body Acc Z'); ax.legend(); ax.set_xlabel('Time (s)'); st.pyplot(fig); fig2,ax2=plt.subplots(figsize=(9,2.5)); ax2.plot(t,att); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Attention'); st.pyplot(fig2)
st.caption('Model scores are illustrative and are not medical, diagnostic, or safety-grade outputs.')
