



import pandas as pd
import os



folder_path = 'data/new_csv'
filenames = os.listdir(folder_path)
print(filenames)

"""co2_y,pleth_y,ecg_y"""
data_II = []
data_PLETH = []
data_RESP = []
len_all=0
for filename in filenames:
    print('\n')
    print(filename)
    
    df = pd.read_csv('data/new_csv/' + filename, header=0)
    data1 = list(df['co2_y'])
    data2 = list(df['pleth_y'])
    data3 = list(df['ecg_y'])
    print(len(df))
    len_all=len_all+len(df)
  
    data_II = data_II + data1
    data_PLETH = data_PLETH + data2
    data_RESP = data_RESP + data3
print(len_all)

time=list(range(len(data_II)))
savedf = pd.DataFrame()
savedf['time'] = time
savedf['co2_y'] = data_II
savedf['pleth_y'] = data_PLETH
savedf['ecg_y'] = data_RESP
# savedf.to_csv(path_or_buf='data\\data_sum.csv', header=True, index=False, encoding='utf-8')
savedf=savedf[::15]
print(len(savedf))
# savedf.to_csv(path_or_buf='data\\ETT\\column_data.csv', header=True, index=False, encoding='utf-8')
savedf.to_csv(path_or_buf='data/data_sum_delete.csv', header=True, index=False, encoding='utf-8')



