
import numpy as np

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
