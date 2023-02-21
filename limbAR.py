from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
#%% read data
def read_data(file_path):
    df = pd.read_csv(file_path, header=[0,1,2])
    acol = [14*i+1 for i in range(6)] + [14*i+2 for i in range(6)] + [14*i+3 for i in range(6)]
    gcol = [14*i+4 for i in range(6)] + [14*i+5 for i in range(6)] + [14*i+6 for i in range(6)]
    col = acol + gcol
    col.sort()
    df_new = df.iloc[:,col]
    df_new.columns = ['LT_a_x', 'LT_a_y', 'LT_a_z', 'LT_g_x', 'LT_g_y', 'LT_g_z',
                    'W_a_x', 'W_a_y', 'W_a_z', 'W_g_x', 'W_g_y', 'W_g_z',
                    'LS_a_x', 'LS_a_y', 'LS_a_z', 'LS_g_x', 'LS_g_y', 'LS_g_z',
                    'RS_a_x', 'RS_a_y', 'RS_a_z', 'RS_g_x', 'RS_g_y', 'RS_g_z',
                    'RT_a_x', 'RT_a_y', 'RT_a_z', 'RT_g_x', 'RT_g_y', 'RT_g_z',
                    'C_a_x', 'C_a_y', 'C_a_z', 'C_g_x', 'C_g_y', 'C_g_z']   
    return df_new

      
def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan

#%% plot confusion matrix
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

def show_confusion_matrix(validations, predictions, LABELS):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#%% sliding window
def creat_segment(df, size, ovlp_ratio, label_name):

    N_FEATURES = 36
    
    segments = []
    labels = []
    i = 0
    
    while size + i <= len(df):
        sw = df.iloc[i : size+i]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: size+i])[0][0]
        sw = sw.drop(columns=label_name)
        segments.append(sw)
        labels.append(label)
        i = int(i + size*(1-ovlp_ratio))

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, size, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

#%% normalization
def normalise(df):
    
    for i in range(0, df.shape[1]):
        df.iloc[:, i] = (df.iloc[:, i]-df.iloc[:, i].min())/(df.iloc[:, i].max()-df.iloc[:, i].min())
    
    return df

#%%
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
LABELS = ['SLR(R)', 'SLR(L)', 'SAE(R)', 'SAE(L)', 'KE(R)', 'KE(L)']
   
def points_validation(df, y_pred, y_gt, size, ovlp):
    i = 1
    df_pred = np.zeros(len(df),)

    df_pred[0:size] = y_pred[0]

    for i in range(1,len(y_pred)):
        df_pred[int((size*(1-ovlp)*i+size+size*(1-ovlp)*(i-1))/2):int(size+size*(1-ovlp)*i)] = y_pred[i]
    
    df_pred = np.delete(df_pred, np.where(df_pred==0), 0)
    
    print(classification_report((y_gt[0:len(df_pred)]).astype(int), df_pred.astype(int)))
    
    matrix = metrics.confusion_matrix(y_gt[0:len(df_pred)], df_pred.astype(int))
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
#%%
from sklearn import metrics
from matplotlib import pyplot as plt
   
def points_prediction(df, y_pred, y_gt, size, ovlp):
    i = 1
    df_pred = np.zeros(len(df),)

    df_pred[0:size] = y_pred[0]

    for i in range(1,len(y_pred)):
        df_pred[int((size*(1-ovlp)*i+size+size*(1-ovlp)*(i-1))/2):int(size+size*(1-ovlp)*i)] = y_pred[i]
    
    df_pred = np.delete(df_pred, np.where(df_pred==0), 0)
        
    return df_pred.astype(int), y_gt[0:len(df_pred)].astype(int)
    
    
    
        
