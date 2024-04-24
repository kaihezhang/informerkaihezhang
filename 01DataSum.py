
# file_list = glob.glob('data\\DataKH\\*.csv')
# print(file_list)
#

# combined_data = pd.DataFrame()
#
# len_all=0

# for file in file_list:
#     df = pd.read_csv(file)
#     print(len(df))
#     len_all=len_all+len(df)
#     combined_data = combined_data.append(df, ignore_index=True)
#

# combined_data.to_csv('combined_data.csv', index=False)
# print(len(combined_data))


import os

import pandas as pd

folder_path = 'data\\DataKH'
filenames = os.listdir(folder_path)
print(filenames)

data_II = []
data_PLETH = []
data_RESP = []
len_all=0
for filename in filenames:
    print('\n')
    print(filename)
    
    df = pd.read_csv('data\\DataKH\\' + filename, header=0)
    data1 = list(df['II'])
    data2 = list(df['PLETH'])
    data3 = list(df['RESP'])
    print(len(df))
    len_all=len_all+len(df)
   
    data_II = data_II + data1
    data_PLETH = data_PLETH  + data2
    data_RESP = data_RESP + data3
print(len_all)

time=list(range(len(data_II)))
savedf = pd.DataFrame()
savedf['time'] = time
savedf['II'] = data_II
savedf['PLETH'] = data_PLETH
savedf['RESP'] = data_RESP
# savedf.to_csv(path_or_buf='data\\data_sum.csv', header=True, index=False, encoding='utf-8')
savedf=savedf[::10]
savedf.to_csv(path_or_buf='data\\ETT\\column_data.csv', header=True, index=False, encoding='utf-8')



