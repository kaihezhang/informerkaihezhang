import pandas as pd
import csv
import datetime

def TimeTrans():
    filepath = 'data\\ETT\\column_data.csv'
    data = pd.read_csv(filepath, header=0)  
    time=list(data['time'])

    start_date = datetime.datetime(2000, 1, 1, 00, 00)
    # print('start_date is :', start_date)
    trained_len = len(time)
    # print('trained_len is :', trained_len)
    Trans_num = trained_len - 1  
    arr_date = []
    arr_date.append(start_date)
    for i in range(Trans_num):
        if i == 0:
            date = start_date + datetime.timedelta(hours=1)
        else:
            date = date + datetime.timedelta(hours=1)
        # print('When No is :',i+1,',date is :',date)     。
        arr_date.append(date)

    print('TimeTrans has been finished\n')

  
    savedf=data
    savedf['time'] = list(arr_date)
    savedf = savedf.rename(columns={'time': 'date'})
    savedf = savedf.rename(columns={'RESP': 'OT'})
    savedf.to_csv(path_or_buf='data\\ETT\\ETTh1.csv', header=True, index=False, encoding='utf-8')

    return None