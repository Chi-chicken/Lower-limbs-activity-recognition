# Lower Limbs Activity Recognition using Machine Learning Methods

This project aims to classify the 3 common lower limbs activities (with both left and right side) of osteoarthritis rehabilitation with 3axes' accelerometers ang gyroscopes. The machine learning models are used to classification.

### Lower limbs activity
  * right-side Straight Leg Raise (SLR-R)
  * left-side Straight Leg Raise (SLR-L)
  * right-side Short-arc Exercise (SAE-R)
  * left-side Short-arc Exercise (SAE-L)
  * right-side Snee Extension (KE-R)
  * left-side Snee Extension (KE-L)

### Sensors
 * 6 OPAL IMUs (published by APDM, Portland, USA)
 * Sampling rate: 128 Hz
 * Sensors(6) location:  Chest, waist, both thighs, and both shanks

## Methods

### Data loading and preprocessing
 * Execute file `limbAR.py`
 * Load data (`read_data`)
 * Normalise
 ```python
 def normalise(df):
    
    for i in range(0, df.shape[1]):
        df.iloc[:, i] = (df.iloc[:, i]-df.iloc[:, i].min())/(df.iloc[:, i].max()-df.iloc[:, i].min())
    
    return df
 ```
 * Sliding window
 ```python
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
  ```
  * Split data into training set and testing set(`lowerlimb_HAR.py`)
```python 
df_train, df_test = train_test_split(df, test_size=0.3)
x_train, y_train = df_train.drop(columns='activity'), df_train['activity']
x_test, y_test = df_test.drop(columns='activity'), df_test['activity']

x_train, y_train = x_train.values, y_train.values  
x_test, y_test = x_test.values, y_test.values  

y_train_hot = np_utils.to_categorical(y_train)

```
