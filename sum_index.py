import os

import pandas as pd

folder_path = 'results/Index'
filenames = os.listdir(folder_path)
print(filenames)
len_filenames = len(filenames)
print('\nlen filenames is {} '.format(len_filenames))

file_list = []
mae_list = []
rmse_list = []
mape_list = []
R2_list = []
count_file = 0
for filename in filenames:
   
    print('\nHere comes {} , {} / {} '.format(filename, count_file, len_filenames))
    count_file = count_file + 1
  
    df = pd.read_csv('results/Index/' + filename, header=0)
    print(df)
    file = filename[-11:-4]
    file_list.append(file)
    mae_list.append(list(df['MAE'])[0])
    rmse_list.append(list(df['RMSE'])[0])
    mape_list.append(list(df['MAPE'])[0])
    R2_list.append(list(df['R2'])[0])

savedf = pd.DataFrame()
savedf['file'] = file_list
savedf['MAE'] = mae_list
savedf['RMSE'] = rmse_list
savedf['MAPE'] = mape_list
savedf['R2'] = R2_list
savedf.to_csv(path_or_buf='results/sum_index.csv', header=True, index=False, encoding='utf-8')
