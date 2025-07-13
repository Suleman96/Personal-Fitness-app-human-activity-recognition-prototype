#imports

import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,GRU,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import pickle 
import os 


#%cd ./drive/MyDrive/EmbeddedCasestudy/CSV
# Using a pandas dataframe to load the data from recordings
#sitting
df_sitting1 = pd.read_csv('/content/Sitting 1.csv')
df_sitting1.head()
df_sitting1.drop(['Time (s)'],axis=1,inplace=True)
dataset_sit1 = np.array(df_sitting1)
x_dataset_sit1=dataset_sit1[1000:-1000,:3]

df_sitting2 = pd.read_csv('/content/Sitting 2.csv')
df_sitting2.head()
df_sitting2.drop(['Time (s)'],axis=1,inplace=True)
dataset_sit2 = np.array(df_sitting2)
x_dataset_sit2=dataset_sit2[1000:-1000,:3]

#walking
df_walking1 = pd.read_csv('/content/Walking 1.csv')
df_walking1.head()
df_walking1.drop(['Time (s)'],axis=1,inplace=True)
dataset_walk1 = np.array(df_walking1)
x_dataset_walk1=dataset_walk1[1000:-1000,:3]

df_walking2 = pd.read_csv('/content/Walking 2.csv')
df_walking2.head()
df_walking2.drop(['Time (s)'],axis=1,inplace=True)
dataset_walk2 = np.array(df_walking2)
x_dataset_walk2=dataset_walk2[1000:-1000,:3]

#running
df_run1 = pd.read_csv('/content/Running 1.csv')
df_run1.head()
df_run1.drop(['Time (s)'],axis=1,inplace=True)
dataset_run1 = np.array(df_run1)
x_dataset_run1=dataset_run1[1000:14000,:3]

df_run2 = pd.read_csv('/content/Running 2.csv')
df_run2.head()
df_run2.drop(['Time (s)'],axis=1,inplace=True)
dataset_run2 = np.array(df_run2)
x_dataset_run2=dataset_run2[1000:-1000,:3]


print(x_dataset_sit1.shape, x_dataset_sit2.shape, x_dataset_walk1.shape, x_dataset_walk2.shape, x_dataset_run1.shape, x_dataset_run2.shape)


scaler = MinMaxScaler()
x_dataset_sit1 = scaler.fit_transform(x_dataset_sit1)
plt.plot(x_dataset_sit1)

x_dataset_sit2 = scaler.fit_transform(x_dataset_sit2)
plt.plot(x_dataset_sit2)

x_dataset_walk1 = scaler.fit_transform(x_dataset_walk1)
plt.plot(x_dataset_walk1)

x_dataset_walk2 = scaler.fit_transform(x_dataset_walk2)
plt.plot(x_dataset_walk2)

x_dataset_run1 = scaler.fit_transform(x_dataset_run1)
plt.plot(x_dataset_run1)

x_dataset_run2 = scaler.fit_transform(x_dataset_run2)
plt.plot(x_dataset_run2)

x_dataset_walk1[2600]

import tensorflow as tf
#getdata function creates data for training gru model
#data is the raw dataset, three acc columns
#datatype 0 = sitting , 1 = walking, 2 = running
# lookback is the sample rate
# interval defines time interval
# fr is the dataset frequency
def getdata(data, datatype, lookback, interval,fr):
  finterval=interval*fr
  period=finterval//(lookback-1) #period= steps of extracing data
  X,Y=[],[]
  for i in range(len(data)-finterval):
    samplelist=[]
    for j in range(lookback):
      samplelist.append(data[i+(j*period),:3])
    X.append(samplelist)
    Y.append(datatype)
    # one hot encoding
  Y = tf.one_hot(Y, 3).numpy()
  return np.array(X),np.array(Y)
lookback=12  
interval=120
fr=100

x_processed_sit1,y_processed_sit1=getdata(x_dataset_sit1, 0,lookback,interval,fr)
x_processed_sit2,y_processed_sit2=getdata(x_dataset_sit2, 0,lookback,interval,fr)
x_processed_walk1,y_processed_walk1=getdata(x_dataset_walk1, 1,lookback,interval,fr)
x_processed_walk2,y_processed_walk2=getdata(x_dataset_walk2, 1,lookback,interval,fr)
x_processed_run1,y_processed_run1=getdata(x_dataset_run1, 2,lookback,interval,fr)
x_processed_run2,y_processed_run2=getdata(x_dataset_run2, 2,lookback,interval,fr)

print(x_processed_sit1.shape,x_processed_sit2.shape,y_processed_walk1.shape,y_processed_walk2.shape,x_processed_run1.shape,x_processed_run2.shape)

X_sit=np.concatenate((x_processed_sit1,x_processed_sit2),axis=0)
X_walk=np.concatenate((x_processed_walk1,x_processed_walk2),axis=0)
X_run=np.concatenate((x_processed_run1,x_processed_run2),axis=0)
print(X_sit.shape,X_walk.shape,X_run.shape)

X_walk[200:210]


#y_sit1 = tf.one_hot(y_processed_sit1, 3).numpy()
#y_sit2 = tf.one_hot(y_processed_sit2, 3).numpy()
Y_sit=np.concatenate((y_processed_sit1,y_processed_sit2),axis=0)

#y_walk1 = tf.one_hot(y_processed_walk1, 3).numpy()
#y_walk2 = tf.one_hot(y_processed_walk2, 3).numpy()
Y_walk=np.concatenate((y_processed_walk1,y_processed_walk2),axis=0)

#y_run1 = tf.one_hot(y_processed_run1, 3).numpy()
#y_run2 = tf.one_hot(y_processed_run2, 3).numpy()
Y_run=np.concatenate((y_processed_run1,y_processed_run2),axis=0)

print(Y_sit.shape, Y_walk.shape, Y_run.shape)




from sklearn.model_selection import train_test_split
X_sit_train, X_sit_test, Y_sit_train, Y_sit_test = train_test_split(X_sit, Y_sit, test_size=0.2, random_state=0)
X_walk_train, X_walk_test, Y_walk_train, Y_walk_test = train_test_split(X_walk, Y_walk, test_size=0.2, random_state=0)
X_run_train, X_run_test, Y_run_train, Y_run_test = train_test_split(X_run, Y_run, test_size=0.2, random_state=0)

print(X_sit_train.shape, X_sit_test.shape, Y_walk_train.shape, Y_walk_test.shape, X_run_train.shape, Y_run_test.shape)

X_train= np.concatenate((X_sit_train, X_walk_train, X_run_train), axis=0)
X_test=np.concatenate((X_sit_test, X_walk_test, X_run_test), axis=0)
Y_train= np.concatenate((Y_sit_train, Y_walk_train, Y_run_train), axis=0)
Y_test=np.concatenate((Y_sit_test, Y_walk_test, Y_run_test), axis=0)


########################## LSTM ####################################################################

model=Sequential()
model.add(LSTM(20,input_shape=(lookback,3)))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train, batch_size=1, epochs=5)

#plt.figure(figsize=(14,5))
#plt.plot(y_test, label = 'y_test')
#plt.plot(y_pred, label = 'y_pred')
#plt.legend()
#plt.show()  

model.fit(X_train, Y_train)

model.evaluate(X_test, Y_test)

# MOVEMENT of user are classified. 3 class are used
ACTIVITIES = {
    0: 'sitting',
    1: 'walking',
    2: 'runing',
}

# Confusion Matrix
def confusion_matrix(Y_true, Y_pred):
    
    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])
    plt.show()
confusion_matrix(Y_test, model.predict(X_test))

y_pred=model.predict(X_test)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])

# path to the SavedModel directory
model.save('/content/drive/MyDrive/DIT /Embedded System/LSTM App/Models')

# path to the SavedModel directory
model = tf.saved_model.load('/content/drive/MyDrive/DIT /Embedded System/LSTM App/Models') 
# tflite_model = converter.convert()

os.chdir('/content/drive/MyDrive/DIT /Embedded System/LSTM App/Models')
with open('Suleman_PKL', 'wb') as files:
    pickle.dump(model, files)

converter = tf.lite.TFLiteConverter.from_saved_model('/content/drive/MyDrive/DIT /Embedded System/LSTM App/Models') 
# path to the SavedModel directory

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()


import os
os.chdir('/content/drive/MyDrive/DIT /Embedded System/LSTM App/Models')

# Save the model.
with open('suleman_model.tflite', 'wb') as f:
  f.write(tflite_model)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



















# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib as tf
import random
import keras
#from matplotlib import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import L1L2
from keras import optimizers
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
# dirname = os.path.dirname(__file__)
def prediction(data):
    ACTIVITIES = {
        0: 'sitting',
        1: 'walking',
        2: 'running',
    }


# q = ([0.65651394, 0.65701814, 0.70651538],
#         [0.65656196, 0.65693411, 0.70640734],
#         [0.65648993, 0.65707817, 0.70640734],
#         [0.65652595, 0.65705416, 0.70647937],
#         [0.65657397, 0.65698213, 0.70649138],
#         [0.65654996, 0.6569101 , 0.70640734],
#         [0.65653795, 0.65701814, 0.70645536],
#         [0.65647793, 0.6568981 , 0.70639534],
#         [0.65651394, 0.65701814, 0.70640734],
#         [0.65654996, 0.65700614, 0.70641935],
#         [0.65663399, 0.65697012, 0.70641935],
#         [0.65658597, 0.65704215, 0.70645536])


# model= tf.lite.Interpreter(model_path="/Models/suleman_model.tflite")

# model.allocate_tensors()


# input_details= model.get_input_details()
# output_details= model.get_output_details()

# input_shape = input_details[0]['shape']
# input_data=np.array(np.random_sample(input_shape), dtype=np.float32)
# model.set-tensot(input_details[0]['index'], input_data)
# model.invoke()


# output_data= model.get_tensor(output_details[0]['index'])
# print(output_data)

# Sensor_values=[]
# for i in range(0, 12):
#     Sensor_values.append([random.random(),random.random(),random.random()])

    model = keras.models.load_model('C:/Users/Lenovo/Desktop/DIT AI Lectures/MSS-M-1 Case Study Embedded Control Solutions (SS22)/FINAL Project/Models')
    predicted_model = model.predict((Sensor_values)) 

    q = ACTIVITIES[np.argmax(predicted_model)]


    return q

