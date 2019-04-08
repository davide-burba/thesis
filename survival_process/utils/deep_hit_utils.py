
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
    def forward(self,y_pred,y,status):
        totloss = 0
        for i in range(len(y)):
            v = y[i]
            k = status[i]
            y_pred_i = y_pred[i,:]
            if k.item() == 1:
                totloss -= torch.log(y_pred_i[v])
            else:
                totloss -= torch.log(1-torch.sum(y_pred_i[:v]))
        return totloss


class ColumnarDataset(Dataset):
    '''Class to manage tabular dataset in pytorch framework
    '''
    def __init__(self, X, y,status):
        self.dfconts = X
        self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values
        self.status = status.values

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.conts[idx], self.y[idx], self.status[idx]]

    
    
def compute_expected_survival_time(predictions,n_times,T_max):
    ''' Function to compute expected survival time for each patient from predictions
    '''
    expected_survival_times = []
    numpy_pred = predictions.detach().numpy()
    for i in range(len(numpy_pred)):
        pred_i = numpy_pred[i,:]
        times = 0.5 + np.arange(len(pred_i))
        # come back to original scale
        times = times/n_times*T_max

        expected_survival_times.append(np.dot(pred_i,times))
    return np.array(expected_survival_times)