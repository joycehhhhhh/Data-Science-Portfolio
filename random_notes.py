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


