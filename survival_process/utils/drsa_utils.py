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
    def __init__(self,n_features,dropout = 0.6):
        super(DRSA, self).__init__()
        self.n_features = n_features
        self.hidden_dim = 10*n_features
        self.lstm = nn.LSTM(n_features, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        # custom learnable initial values for h0 and c0
        self.initialise_h0 = nn.Linear(self.n_features-1,self.hidden_dim)
        self.initialise_c0 = nn.Linear(self.n_features-1,self.hidden_dim)

    def forward(self, x):
        # compute initial values hidden and cell vectors
        h0 = torch.relu(self.initialise_h0(x[0,:,:self.n_features-1]).reshape(1,-1,self.hidden_dim))
        c0 = torch.relu(self.initialise_c0(x[0,:,:self.n_features-1]).reshape(1,-1,self.hidden_dim))
        h0 = self.dropout(h0)
        c0 = self.dropout(c0)
        # run model recurrently for the whole sequences
        x = self.dropout(self.lstm(x,(h0,c0))[0])
        x = self.output(x)
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
        tmp = y[mask & (y != 0)]-1
        sum_one_minus_log = torch.log(1-y_pred[mask & (y != 0)]).cumsum(1).gather(1,tmp.view(-1,1))
        Lz = -(logs.sum() + sum_one_minus_log.sum())

        # Compute L_uncensored
        L_uncensored = -torch.log(1-(1-y_pred[mask]).cumprod(1).gather(1,y[mask].view(-1,1))).sum()

        # Compute L_censored
        L_censored = -torch.log(1-y_pred[1-mask]).cumsum(1).gather(1,y[1-mask].view(-1,1)).sum()

        Lc = L_uncensored +L_censored

        loss = self.alpha*Lz+(1-self.alpha)*Lc

        return loss



def reformat_dataset(x_in,times):
    ''' Adds time feature, returns array with 3 dimensions: time,id,features
    '''
    x_out = []
    for i in x_in.index:
        tmp = np.repeat(x_in.loc[i,:].values.reshape(1,1,-1),len(times),axis = 0)
        observation_i = np.concatenate([tmp,times], axis = 2)
        x_out.append(observation_i)
    x_out = np.concatenate(x_out, axis = 1)
    return x_out


def get_survival_density(predictions):
    '''compute survival time probabilities from hazards'''
    survival_density = [predictions[:,0].reshape(-1,1)]
    tmp = (1 - predictions).cumprod(1)
    for j in np.arange(1,predictions.shape[1]):
        if j > 0:
            survival_density.append((predictions[:,j]*tmp[:,j-1]).reshape(-1,1))
    survival_density = torch.cat(survival_density,dim = 1)
    return survival_density
