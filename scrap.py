from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from keras.layers import Dropout
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import constants

# creates the tensorboard log
NAME = "test{}".format(int(time.time())) #TODO change to class and x is an input var
tensorboard = TensorBoard(log_dir="tensorboard_log/{}".format(NAME))

#connect to db
db = constants.DATABASES['local']
engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user =      db['USER'],
        password =  db['PASSWORD'],
        host =      db['HOST'],
        port =      db['PORT'],
        database =  db['NAME']
    )

#pull in the data and format it by taking the mean between the asking and bid price
engine = create_engine(engine_string)
data = pd.read_sql_table('usd_jpy',engine) #TODO change table name to a var
df = pd.DataFrame
df = data[['date']].copy()
df['open'] =  data[['bidopen', 'askopen']].mean(axis=1)
df['close'] = data[['bidclose', 'askclose']].mean(axis=1)
df['high'] = data[['bidhigh', 'askhigh']].mean(axis=1)
df['low'] = data[['bidlow', 'asklow']].mean(axis=1)

transformedDataSet = pd.DataFrame(columns=["o5","c5","h5","l5","o4","c4","h4","l4","o3","c3","h3","l3","o2","c2","h2","l2","o1","c1","h1","l1","actual_open","actual_close","actual_high","actual_low"])

#convert the data frame into data sets for training and testing
for index, row in df.iterrows():
    if index >= 5:
        transformedDataSet.loc[index-5] = [
            df.loc[index-5,"open"],df.loc[index-5,"close"],df.loc[index-5,"high"],df.loc[index-5,"low"],
            df.loc[index-4,"open"],df.loc[index-4,"close"],df.loc[index-4,"high"],df.loc[index-4,"low"],
            df.loc[index-3,"open"],df.loc[index-3,"close"],df.loc[index-3,"high"],df.loc[index-3,"low"],
            df.loc[index-2,"open"],df.loc[index-2,"close"],df.loc[index-2,"high"],df.loc[index-2,"low"],
            df.loc[index-1,"open"],df.loc[index-1,"close"],df.loc[index-1,"high"],df.loc[index-1,"low"],
            df.loc[index,"open"],df.loc[index,"close"],df.loc[index,"high"],df.loc[index,"low"]
        ]

#break apart the dataframe into training and testing data frames and inputs and outputs
print("Full data set size")
print(len(transformedDataSet.index))
print()

msk = np.random.rand(len(transformedDataSet)) < 0.8
x_train = transformedDataSet[msk]
x_test = transformedDataSet[~msk]

y_train = x_train[["actual_open","actual_close","actual_high","actual_low"]].reset_index(drop=True)
y_test = x_test[["actual_open","actual_close","actual_high","actual_low"]].reset_index(drop=True)
x_train = x_train[["o5","c5","h5","l5","o4","c4","h4","l4","o3","c3","h3","l3","o2","c2","h2","l2","o1","c1","h1","l1"]].reset_index(drop=True)
x_test = x_test[["o5","c5","h5","l5","o4","c4","h4","l4","o3","c3","h3","l3","o2","c2","h2","l2","o1","c1","h1","l1"]].reset_index(drop=True)

#data frame of values to be predicted
x_pred2 = y_test

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

print("training size then test size")
print(x_train.shape)
print(x_test.shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4))

model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics =['accuracy'])



model.fit(x_train, y_train, epochs=20, callbacks=[tensorboard], validation_data=(x_test,y_test))

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

prediction = model.predict(x_test,verbose=1)
prediction = pd.DataFrame(data=prediction)


print("prediction size")
print(len(prediction))
