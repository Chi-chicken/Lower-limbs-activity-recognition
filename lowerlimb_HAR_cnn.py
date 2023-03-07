#%%1
from __future__ import print_function
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats


from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import limbAR as ar 
#%%2
df_1 = ar.read_data(r'.\20200707-183957-subject_1.csv')
df_2 = ar.read_data(r'.\20200720-183906-subject_2.csv')
df_3 = ar.read_data(r'.\20200720-191919-subject_3.csv')
df_4 = ar.read_data(r'.\20200720-195621-subject_4.csv')
df_5 = ar.read_data(r'.\20200720-204112-subject_5.csv')

df_1 = ar.normalise(df_1)
df_2 = ar.normalise(df_2)
df_3 = ar.normalise(df_3)
df_4 = ar.normalise(df_4)
df_5 = ar.normalise(df_5)

seg_1 = np.array([3620, 30702, 42332, 68296, 77791, 103831, 111332, 137167, 205678, 231569, 168282, 194433])
seg_2 = np.array([2707, 28457, 34733, 61969, 70499, 96813, 103063, 130119, 147282, 173440, 177340, 204503])
seg_3 = np.array([2464, 28828, 35740, 62720, 70700, 97235, 101454, 127456, 142534, 168434, 173882, 198753])
seg_4 = np.array([10579, 36517, 49085, 75425, 86260, 113206, 116545, 142555, 157037, 184093, 193652, 220550])
seg_5 = np.array([1700, 28858, 32009, 58195, 67246, 94376, 98560, 124502, 135355, 162573, 167923, 194271])

activityCode = [0, 1, 2, 3, 4, 5]


for i in range(0,11,2):
    df_1.loc[seg_1[i]:seg_1[i+1], 'activity'] = i/2+1
for i in range(0,11,2):
    df_2.loc[seg_2[i]:seg_2[i+1], 'activity'] = i/2+1
for i in range(0,11,2):
    df_3.loc[seg_3[i]:seg_3[i+1], 'activity'] = i/2+1
for i in range(0,11,2):
    df_4.loc[seg_4[i]:seg_4[i+1], 'activity'] = i/2+1
for i in range(0,11,2):
    df_5.loc[seg_5[i]:seg_5[i+1], 'activity'] = i/2+1

df_1 = df_1.dropna(how='any')
df_2 = df_2.dropna(how='any')
df_3 = df_3.dropna(how='any')
df_4 = df_4.dropna(how='any')
df_5 = df_5.dropna(how='any')


#%%3
size = 1280
ovlp_ratio = 0.9
x_1, y_1 = ar.creat_segment(df_1, size, ovlp_ratio, 'activity')
x_2, y_2 = ar.creat_segment(df_2, size, ovlp_ratio, 'activity')
x_3, y_3 = ar.creat_segment(df_3, size, ovlp_ratio, 'activity')
x_4, y_4 = ar.creat_segment(df_4, size, ovlp_ratio, 'activity')
x_5, y_5 = ar.creat_segment(df_5, size, ovlp_ratio, 'activity')

x_1 = x_1.reshape(x_1.shape[0], -1)
x_2 = x_2.reshape(x_2.shape[0], -1)
x_3 = x_3.reshape(x_3.shape[0], -1)
x_4 = x_4.reshape(x_4.shape[0], -1)
x_5 = x_5.reshape(x_5.shape[0], -1)

y_1 = y_1.reshape(y_1.shape[0], -1)
y_2 = y_2.reshape(y_2.shape[0], -1)
y_3 = y_3.reshape(y_3.shape[0], -1)
y_4 = y_4.reshape(y_4.shape[0], -1)
y_5 = y_5.reshape(y_5.shape[0], -1)

s1 = np.concatenate((x_1, y_1), axis=1)
s2 = np.concatenate((x_2, y_2), axis=1)
s3 = np.concatenate((x_3, y_3), axis=1)
s4 = np.concatenate((x_4, y_4), axis=1)
s5 = np.concatenate((x_5, y_5), axis=1)

np.random.shuffle(s1)
np.random.shuffle(s2)
np.random.shuffle(s3)
np.random.shuffle(s4)
np.random.shuffle(s5)

x_1 = s1[:,:size*36]
y_1 = s1[:, size*36]

x_2 = s2[:,:size*36]
y_2 = s2[:, size*36]

x_3 = s3[:,:size*36]
y_3 = s3[:, size*36]

x_4 = s4[:,:size*36]
y_4 = s4[:, size*36]

x_5 = s5[:,:size*36]
y_5 = s5[:, size*36]

x = [x_1, x_2, x_3, x_4, x_5]
y = [np_utils.to_categorical(y_1-1), np_utils.to_categorical(y_2-1), np_utils.to_categorical(y_3-1), np_utils.to_categorical(y_4-1), np_utils.to_categorical(y_5-1)]

#%%4
model = Sequential()
# 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size,36,1)))
# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
#model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(128, activation='relu'))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
#model.add(Dropout(0.5))
# 使用 softmax activation function，將結果分類
model.add(Dense(6, activation='softmax'))

print(model.summary())

BATCH_SIZE = 50
EPOCHS = 16
#%%5
cv = LeaveOneOut()
#cv.get_n_splits(df_x_train)

for train_index, cv_index in cv.split(x):
    print("TRAIN:", train_index, "TEST:", cv_index)
    x_train = np.concatenate(np.array([x[ii] for ii in train_index]))
    y_train = np.concatenate(np.array([y[ii] for ii in train_index]))

    x_test = np.concatenate(np.array([x[ii] for ii in cv_index]))
    y_test = np.concatenate(np.array([y[ii] for ii in cv_index]))
    
    #cross-entropy lower，loss lower, model better
    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    
    
    x_train = x_train.reshape(x_train.shape[0], size, 36, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], size, 36, 1).astype('float32')
    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    # verbose 0 = silent, 1 = progress bar, 2 = one line per epoch
    # validation_split Fraction of the training data to be used as validation data
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3, verbose=1)
    
    # Print confusion matrix for training data
    y_pred_test = model.predict(x_test)
    # Take the class with the highest probability from the train predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)  
    print(classification_report(np.argmax(y_test, axis=1), max_y_pred_test))
    print(confusion_matrix(np.argmax(y_test, axis=1), max_y_pred_test))
    #np.savetxt(("y_predict_cnn_"+str(cv_index[0]+1)+".csv"), max_y_pred_test, delimiter=',')










