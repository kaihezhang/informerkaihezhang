from models.TimeTrans import TimeTrans
from My_Informer import My_Informer
from models.My_Visual import My_Visual
import os

import pandas as pd

from models.My_Index import My_Index

train_set = 0.8
savedf = pd.DataFrame()
savedf['method'] = ['Informer']
savedf['train_set'] = [str(train_set)]
print('\nparameters is : \n', savedf)
savedf.to_csv(path_or_buf='parameters\\parameters.csv', header=True, index=False, encoding='utf-8')

df=pd.read_csv('data\\data_sum.csv', header=0)
df=df[::2]
df.to_csv(path_or_buf='data\\ETT\\column_data.csv', header=True, index=False, encoding='utf-8')


filepath = 'results\\forecast_' + str(train_set)
if not os.path.exists(filepath):
    os.mkdir(filepath)


TimeTrans()
My_Informer()
My_Visual()
My_Index()
