#standardize the data
cols_to_scale=['column1','column2']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler
df1[cols_to_scale]=scaler.fit_transform(df1[cols_to_scale])
df.sample(3)

#one hot coding for data preparation
df2=pd.get_dummies(data=df1, columns=['column1', 'column2'])
df2.columns

#see unique values in each column
for col in df2:
	print(f'{col}:{df1[col].unique()'})

#prepare to do the analysis--Artificial Neural Network (ANN)
X=df2.drop('Churn', axis='columns')
y=df2['Churn']

#the analysis--Artificial Neural Network (ANN)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=5)
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
#opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test, y_test)
yp = model.predict(X_test)
yp[:5]
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred[:10]

#prediction result
from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test,y_pred))
import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
