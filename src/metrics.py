import numpy as np

def accuracy(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); return float(np.mean(y_true==y_pred))

def confusion_matrix(y_true,y_pred,num_classes=6):
    cm=np.zeros((num_classes,num_classes),dtype=np.int64)
    for t,p in zip(np.asarray(y_true).astype(int),np.asarray(y_pred).astype(int)): cm[t,p]+=1
    return cm

def per_class_accuracy(cm):
    return np.divide(np.diag(cm),cm.sum(axis=1),out=np.zeros(cm.shape[0],dtype=float),where=cm.sum(axis=1)!=0)
