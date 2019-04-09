
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    '''Class to manage 3D dataset in pytorch framework for RNN (dimensions: time,id,features)
    '''
    def __init__(self, obs, y,status):
        self.conts = obs.astype(np.float32)
        self.y = y.values
        self.status = status.values.astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.conts[:,idx,:], self.y[idx], self.status[idx]]


class DRSA(nn.Module):
    '''DRSA network architecture
    '''
    def __init__(self,n_features):
        super(DRSA, self).__init__()
        self.lstm = nn.LSTM(n_features, 10*n_features)
        self.output = nn.Linear(10*n_features, 1)
        self.droput = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.droput(self.lstm(x)[0])
        x = self.droput(self.output(x))
        x = torch.sigmoid(x)
        return x


class DRSA_Loss(torch.nn.Module):
    '''DRSA loss function
    '''
    def __init__(self,alpha = 0.25):
        super(DRSA_Loss,self).__init__()
        self.alpha = alpha

    def forward(self,y_pred,y,status):
        mask = (status ==1)

        # Compute Lz
        # compute logs and sum of (1-log)
        logs =  torch.log(y_pred[mask].gather(1,y[mask].view(-1,1)))
        # little trick for events happening at time 0, to avoid looking at index -1
        tmp = y[mask]-1
        tmp[tmp<0]=0
        sum_one_minus_log = torch.log(1-y_pred[mask]).cumsum(1).gather(1,tmp.view(-1,1))
        sum_one_minus_log[y[mask] < 1] = 0

        Lz = -(logs + sum_one_minus_log).sum()

        # Compute L_uncensored
        L_uncensored = -torch.log(1-(1-y_pred[mask]).cumprod(1).gather(1,y[mask].view(-1,1))).sum()

        # Compute L_censored
        L_censored = -torch.log(1-y_pred[1-mask]).cumsum(1).gather(1,y[1-mask].view(-1,1)).sum()

        Lc = L_uncensored+L_censored
        loss = self.alpha*Lz+(1-self.alpha)*Lc

        return loss