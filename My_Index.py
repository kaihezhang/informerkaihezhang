import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def My_Index():
    
    paras = pd.read_csv('parameters\\parameters.csv', header=0)
    train_set = float(paras['train_set'].values[0])
    
    filepath = 'results\\forecast_' + str(train_set) + '\\forecast.csv'
    df = pd.read_csv(filepath, header=0)
    # print(df)
    real = list(df['real'])
    pred = list(df['pred'])

    
    plt.figure(figsize=(12, 8))
    plt.title(filepath)
    plt.plot(real, color='blue', label='Real', linewidth=0.5)
    plt.plot(pred, color='red', label='Pred', linewidth=0.5)
    plt.legend()
    plt.savefig('images\\forecast_' + str(train_set) + '.png', dpi=1000)
    # plt.show()
    plt.close()
    print('\nHere comes draw forecast image ')
    
    mae_list = []
    rmse_list = []
    mape_list = []
    R2_list = []
    # sigam_mean_list = []
    real = df['real']
    pred = df['pred']
    mae = np.mean(np.abs(pred - real))
    rmse = np.sqrt(np.mean((pred - real) ** 2))
    mape = np.mean(np.abs((pred - real) / real)) * 100
    R2 = 1 - np.sum((real - pred) ** 2) / np.sum((real - np.mean(real)) ** 2)
    # sigma_mean = sum(sigma) / len(sigma)
   
    mae_list.append(mae)
    rmse_list.append(rmse)
    mape_list.append(mape)
    R2_list.append(R2)
    # sigam_mean_list.append(sigma_mean)

    
    savedf = pd.DataFrame()
    savedf['MAE'] = mae_list
    savedf['RMSE'] = rmse_list
    savedf['MAPE'] = mape_list
    savedf['R2'] = R2_list
    # savedf['sigma_mean'] = sigam_mean_list
    filepath = 'results\\forecast_index.csv'
    savedf.to_csv(path_or_buf=filepath, header=True, index=False, encoding='utf-8')

    print('\nHere comes index calculation ')

    return None
