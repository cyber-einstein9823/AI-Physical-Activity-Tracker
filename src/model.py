import torch
from torch import nn
from layers import BiLSTM, Attention

class HARNet(nn.Module):
    def __init__(self,input_size=9,hidden_size=64,num_classes=6):
        super().__init__(); self.bilstm=BiLSTM(input_size,hidden_size); self.attention=Attention(hidden_size*2,64); self.classifier=nn.Linear(hidden_size*2,num_classes); bound=1/(hidden_size**0.5); nn.init.uniform_(self.classifier.weight,-bound,bound); nn.init.uniform_(self.classifier.bias,-bound,bound)
    def forward(self,x):
        H=self.bilstm(x); context,alpha=self.attention(H); return self.classifier(context),alpha
if __name__=='__main__':
    torch.manual_seed(0); model=HARNet(); print('Trainable parameters:',sum(p.numel() for p in model.parameters())); x=torch.randn(3,128,9); logits,alpha=model(x); print('Shapes:',logits.shape,alpha.shape,'attention sums:',alpha.sum(dim=1))
