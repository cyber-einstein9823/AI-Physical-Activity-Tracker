import math
import torch
from torch import nn

class RNNCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.input_size=input_size; self.hidden_size=hidden_size
        self.W=nn.Parameter(torch.empty(hidden_size,input_size)); self.U=nn.Parameter(torch.empty(hidden_size,hidden_size)); self.b=nn.Parameter(torch.empty(hidden_size)); self.reset_parameters()
    def reset_parameters(self):
        bound=1/math.sqrt(self.hidden_size)
        for p in self.parameters(): nn.init.uniform_(p,-bound,bound)
    def forward(self,x,h): return torch.tanh(x@self.W.T+h@self.U.T+self.b)

class LSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.input_size=input_size; self.hidden_size=hidden_size
        self.W=nn.Parameter(torch.empty(4*hidden_size,input_size)); self.U=nn.Parameter(torch.empty(4*hidden_size,hidden_size)); self.b=nn.Parameter(torch.empty(4*hidden_size)); self.reset_parameters()
    def reset_parameters(self):
        bound=1/math.sqrt(self.hidden_size)
        for p in self.parameters(): nn.init.uniform_(p,-bound,bound)
    def forward(self,x,h,c):
        gates=x@self.W.T+h@self.U.T+self.b
        i,f,g,o=gates.chunk(4,dim=-1)
        i=torch.sigmoid(i); f=torch.sigmoid(f); g=torch.tanh(g); o=torch.sigmoid(o)
        c_new=f*c+i*g; h_new=o*torch.tanh(c_new)
        return h_new,c_new

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.cell=LSTMCell(input_size,hidden_size); self.hidden_size=hidden_size
    def forward(self,x):
        B,T,_=x.shape; h=x.new_zeros(B,self.hidden_size); c=x.new_zeros(B,self.hidden_size); outputs=[]
        for t in range(T): h,c=self.cell(x[:,t,:],h,c); outputs.append(h)
        return torch.stack(outputs,dim=1)

class BiLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__(); self.forward_lstm=LSTM(input_size,hidden_size); self.backward_lstm=LSTM(input_size,hidden_size)
    def forward(self,x):
        fw=self.forward_lstm(x); bw=self.backward_lstm(torch.flip(x,dims=[1])); bw=torch.flip(bw,dims=[1]); return torch.cat([fw,bw],dim=-1)

class Attention(nn.Module):
    def __init__(self,input_size,attention_size=64):
        super().__init__(); self.Wa=nn.Parameter(torch.empty(attention_size,input_size)); self.ba=nn.Parameter(torch.empty(attention_size)); self.v=nn.Parameter(torch.empty(attention_size)); self.reset_parameters()
    def reset_parameters(self):
        bound=1/math.sqrt(self.Wa.shape[0])
        for p in self.parameters(): nn.init.uniform_(p,-bound,bound)
    def forward(self,H):
        e=torch.tanh(H@self.Wa.T+self.ba)@self.v
        alpha=torch.softmax(e,dim=1); context=torch.sum(alpha.unsqueeze(-1)*H,dim=1)
        return context,alpha
