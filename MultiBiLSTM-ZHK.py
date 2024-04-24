import math
import warnings

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

filepath='data/data_sum_delete_BIDMC.csv'
df=pd.read_csv(filepath)
# print(df)
# df=df[::10]
# print(df)
df.to_csv(path_or_buf='data/new.csv',header=True,index=False,encoding='utf-8')

filepath = 'data/new.csv'
df = pd.read_csv(filepath, parse_dates=["time"], index_col=[0], encoding='utf-8')
# print(df)
len_column = df.shape[1]

dataframe = pd.read_csv(filepath, header=0, parse_dates=[0], index_col=0, usecols=[0, 1]).squeeze('columns')
dataset = dataframe.values
# print(dataframe)

test_split = round(len(df) * 0.20)
df_for_training = df[:-test_split]
df_for_testing = df[-test_split:]
print(df_for_training.shape)
print(df_for_testing.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
training_scaled = scaler.fit_transform(df_for_training)
testing_scaled = scaler.transform(df_for_testing)


def create(dataset, look_back):
    dataX = []
    dataY = []
    for i in range(look_back, len(dataset)):
        dataX.append(dataset[i - look_back:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


look_back = 10
trainX, trainY = create(training_scaled, look_back)
testX, testY = create(testing_scaled, look_back)

model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add((LSTM(30, return_sequences=False)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mse', optimizer='adam')
history = model.fit(trainX, trainY, batch_size=16, epochs=1, validation_split=None, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

prediction_copies_array = np.repeat(testPredict, len_column, axis=-1)
pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(testPredict), len_column)))[:, 0]

original_copies_array = np.repeat(testY, len_column, axis=-1)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), len_column)))[:, 0]


savedf = pd.DataFrame()
savedf['real'] = original.tolist()
savedf['pred'] = pred.tolist()
savedf.to_csv(path_or_buf='results/forecast.csv', header=True, index=False, encoding='utf-8')

print(1)

from My_Index import My_Index
My_Index()
