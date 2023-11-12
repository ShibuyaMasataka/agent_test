import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAE(y_true, y_pred):
    #return np.mean(abs(y_true-y_pred))
    return mean_squared_error(y_true, y_pred)


