
# ! pip install wwo-hist

# from wwo_hist import retrieve_hist_data

# import os
# os.chdir('/content/drive/My Drive/Untitled Folder')

# FREQUENCY = 3
# START_DATE = '11-DEC-2018'
# END_DATE = '11-MAR-2019'
# API_KEY= 'ed802abb3d3a42f7a5e164105202208'
# LOCATION_LIST = ['tehran', 'iran']
# hist_weather_data =  retrieve_hist_data(API_KEY,
#                                         LOCATION_LIST,
#                                         START_DATE,
#                                         END_DATE,
#                                         FREQUENCY,
#                                         location_label = False,
#                                         export_csv = True,
#                                         store_df = True)

import pandas as pd

data =  pd.read_csv('/content/drive/My Drive/Untitled Folder/tehran.csv')
data

data.info()

data.describe()

df = data[['tempC']]

df

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



train_split= 0.9
split_idx = int(len(df) * 0.9)
training_set = df[:split_idx].values
test_set = df[split_idx:].values

training_set.shape

test_set.shape

# 5-day prediction using 30 days data
x_train = []
y_train = []
n_future = 7 #Next 5 days trmpreature forecast
n_past = 30 #Past 30 days
for i in range(0, len(training_set) - n_past - n_future + 1):
    x_train.append(training_set[i : i + n_past, 0])
    y_train.append(training_set[i + n_past : i + n_past + n_future, 0])

x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1))
x_train.shape

import tensorflow as tf

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional


regressor = Sequential()
regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1], 1))))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = n_future, activation='relu'))
regressor.compile( optimizer='adam',
loss='mean_squared_error',
metrics=['acc'])
regressor.fit(x_train, y_train, epochs=500, batch_size=32)

x_test = test_set[: n_past, 0]
y_test = test_set[n_past : n_past + n_future, 0]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
predicted_temperature = regressor.predict(x_test)
print('Predicted temperature {}'.format(predicted_temperature))
print('Real temperature {}'.format(y_test))

plt.figure(figsize = (8,4) )
plt.plot(predicted_temperature[0], label = 'predicted_temperature')
plt.plot(y_test, label = 'y_test')
plt.legend()