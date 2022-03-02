import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

df=pd.read_csv('demo.csv')
df['CY']=df['CY'].astype('object')
df_clean=df[['ULTIMATE_PARENT_ACCOUNT_NAME','CY','AMOUNT_USD_INVOICED']]
df_grp=df_clean.groupby(['ULTIMATE_PARENT_ACCOUNT_NAME', 'CY'], as_index=True).sum()
df_grp.reset_index (inplace=True) 
df_19=df_grp.loc[df_grp['CY']==2019]
df_20=df_grp.loc[df_grp['CY']==2020]
df_21=df_grp.loc[df_grp['CY']==2021]

df20_join = pd.merge(df_19, 
                      df_20, 
                      on ='ULTIMATE_PARENT_ACCOUNT_NAME', 
                      how ='outer')
df21_join = pd.merge(df_20, 
                      df_21, 
                      on ='ULTIMATE_PARENT_ACCOUNT_NAME', 
                      how ='outer')
df20_join['Status_20']='pending'
df21_join['Status_21']='pending'
df20_join["AMOUNT_USD_INVOICED_y"] = df20_join["AMOUNT_USD_INVOICED_y"].fillna(0)
df21_join["AMOUNT_USD_INVOICED_y"] = df21_join["AMOUNT_USD_INVOICED_y"].fillna(0)
for idx, row in df20_join.iterrows():
    if row['AMOUNT_USD_INVOICED_y'] == 0:
        df20_join.loc[idx, 'Status_20'] = 0
    if row['AMOUNT_USD_INVOICED_y'] != 0:
        df20_join.loc[idx, 'Status_20'] = 1
for idx, row in df21_join.iterrows():
    if row['AMOUNT_USD_INVOICED_y'] == 0:
        df21_join.loc[idx, 'Status_21'] = 0
    if row['AMOUNT_USD_INVOICED_y'] != 0:
        df21_join.loc[idx, 'Status_21'] = 1
df20_join['Usage']=np.random.randint(100000, size=6056)
df21_join['Usage']=np.random.randint(100000, size=6142)
df20 = pd.DataFrame({"Parent Account": df20_join['ULTIMATE_PARENT_ACCOUNT_NAME'],
        "Status": df20_join['Status_20'], "Usage": df20_join['Usage']})

df21 = pd.DataFrame({"Parent Account": df21_join['ULTIMATE_PARENT_ACCOUNT_NAME'],
        "Status": df21_join['Status_21'], "Usage": df21_join['Usage']})
frames = [df20, df21]
df1= pd.concat(frames)
df1['Status']=df1['Status'].astype('int32')
cols_to_scale = ['Usage']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])
X = df1.drop('Status',axis='columns')
y = df1['Status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
yp = model.predict(X_test)
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
        
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test,y_pred))

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
