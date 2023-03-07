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

seg_1 = np.array([3620, 30702, 42332, 68296, 77791, 103831, 111332, 137167, 205678, 231569, 168282, 194433])
seg_2 = np.array([2707, 28457, 34733, 61969, 70499, 96813, 103063, 130119, 147282, 173440, 177340, 204503])
seg_3 = np.array([2464, 28828, 35740, 62720, 70700, 97235, 101454, 127456, 142534, 168434, 173882, 198753])
seg_4 = np.array([10579, 36517, 49085, 75425, 86260, 113206, 116545, 142555, 157037, 184093, 193652, 220550])
seg_5 = np.array([1700, 28858, 32009, 58195, 67246, 94376, 98560, 124502, 135355, 162573, 167923, 194271])
activityCode = [0, 1, 2, 3, 4, 5]

df_1['activity'] = 0
df_2['activity'] = 0
df_3['activity'] = 0
df_4['activity'] = 0
df_5['activity'] = 0

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
#%%3
df = df_1.append(df_2).append(df_3).append(df_4).append(df_5)
#shuffle data fran=fraction of data to be return(1->all)
df = df.sample(frac=1)
df_code = df['activity']
df = df.drop(columns='activity')

#normalised
for i in range(0, df.shape[1]):
    df.iloc[:, i] = (df.iloc[:, i]-df.iloc[:, i].min())/(df.iloc[:, i].max()-df.iloc[:, i].min())

df['activity'] = df_code.iloc[:]    
#%%4
df_train, df_test = train_test_split(df, test_size=0.3)
x_train, y_train = df_train.drop(columns='activity'), df_train['activity']
x_test, y_test = df_test.drop(columns='activity'), df_test['activity']

x_train, y_train = x_train.values, y_train.values  
x_test, y_test = x_test.values, y_test.values  

y_train_hot = np_utils.to_categorical(y_train)
#%%5
model = Sequential()
model.add(Dense(100, input_shape=(x_test.shape[1],)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Flatten())
#output layer
model.add(Dense(7, activation='softmax'))

print(model.summary())
#%%6
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

#cross-entropy lower，loss lower, model better
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
# batch size-how amny
# epochs-how many times
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
# verbose 0 = silent, 1 = progress bar, 2 = one line per epoch
# validation_split Fraction of the training data to be used as validation data
history = model.fit(x_train,
                      y_train_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.3,
                      verbose=1)
#%%7
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_train, max_y_pred_train))
#%%8
y_pred_test = model.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
#max_y_test = np.argmax(y_test, axis=1)
LABELS = ['Non', 'SLR(R)', 'SLR(L)', 'SAE(R)', 'SAE(L)', 'KE(R)', 'KE(L)']
ar.show_confusion_matrix(y_test, max_y_pred_test, LABELS)
print(classification_report(y_test, max_y_pred_test))

