
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class DeepHit(nn.Module):
    '''DeepHit network architecture
    '''
    def __init__(self,n_features,n_times):
        super(DeepHit, self).__init__()
        self.fc1 = nn.Linear(n_features, 3*n_features)
        self.fc2 = nn.Linear(3*n_features, 5*n_features)
        self.fc3 = nn.Linear(5*n_features, 3*n_features)
        self.output = nn.Linear(3*n_features, n_times)
        self.droput = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.droput(F.relu(self.fc1(x)))
        x = self.droput(F.relu(self.fc2(x)))
        x = self.droput(F.relu(self.fc3(x)))

        x = self.droput(self.output(x))
        x = F.softmax(x,dim=1)
        return x


class Surv_Loss(torch.nn.Module):
    '''Survival loss function
    '''
    def __init__(self,sigma = 1e2, alpha = 1):
        super(Surv_Loss,self).__init__()
        self.sigma = sigma
        self.alpha = alpha

    def forward(self,y_pred,y,status):
        L1,L2  = 0,0
        F1 = y_pred.cumsum(1)

        # compute L1
        #for uncensored: log P(T=t|x)
        L1 -= torch.log(y_pred.gather(1, y.view(-1,1))).reshape(-1).dot(status)
        #for censored: log \sum P(T>t|x)
        L1 -= torch.log(1-F1.gather(1, y.view(-1,1))).reshape(-1).dot(1-status)

        # compute L2
        for i in range(len(y)):
            if status[i] ==1:
                mask = (y[i] < y) & (status ==1)
                if sum(mask)>0:
                    diff = F1[i,y[i]]-F1[mask,y[i]]
                    L2 += self.alpha*torch.exp(-diff/self.sigma).sum()

        loss = L1 + L2
        return loss


class ColumnarDataset(Dataset):
    '''Class to manage tabular dataset in pytorch framework
    '''
    def __init__(self, X, y,status):
        self.dfconts = X
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values
        self.status = status.values.astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.conts[idx], self.y[idx], self.status[idx]]
