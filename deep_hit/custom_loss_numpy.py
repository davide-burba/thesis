import numpy as np

# fake data
y = np.random.choice(np.arange(20),100)
status = np.random.choice([0,1],100)

# compute fake prediction
y_pred = np.random.uniform(size = (100,20))
y_pred = np.array([v/sum(v) for v in y_pred])

def custom_loss(y_pred,y,status):
    loss = 0
    for i in range(len(y)):
        v = y[i]
        k = status[i]
        y_pred_i = y_pred[i,:]
        loss += _get_contribute_loss(k,v,y_pred_i)
    return loss

_get_contribute_loss = lambda k,v,y_pred_i: -np.log(y_pred_i[v]) if k == 1 else -np.log(1-sum(y_pred_i[:v]))

print(y.shape,status.shape,y_pred.shape)
print('Loss value:',custom_loss(y_pred,y,status))
