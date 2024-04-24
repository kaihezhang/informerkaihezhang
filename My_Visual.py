import os

import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def My_Visual():
    
    paras = pd.read_csv('parameters\\parameters.csv', header=0)
    train_set = float(paras['train_set'].values[0])

    
    preds = np.load('results\\Auto\\pred.npy')
    trues = np.load('results\\Auto\\true.npy')
    new_real = []
    new_pred = []
    for i in range(preds.shape[0]):
        new_real.append(trues[i][0][0])
        new_pred.append(preds[i][0][0])

    
    savedf = pd.DataFrame()
    savedf['real'] = list(new_real)
    savedf['pred'] = list(new_pred)
    savedf.to_csv(path_or_buf='results\\forecast_' + str(train_set) + '\\forecast.csv', header=True, index=False,
                  encoding='utf-8')

    return None
