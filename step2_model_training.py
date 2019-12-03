import csv
import os, sys, email
import numpy as np 
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf



#loading the data
df_goods_features = pd.read_csv('df_goods_features.csv')
df_goods_verbs = pd.read_csv('df_goods_verbs.csv')
df_goods_words = pd.read_csv('df_goods_words.csv')
df_bads_features = pd.read_csv('df_bads_features.csv')
df_bads_verbs = pd.read_csv('df_bads_verbs.csv')
df_bads_words = pd.read_csv('df_bads_words.csv')

# creating the target class value list
goods_flag = pd.DataFrame([1]*len(df_goods_features),columns=['Flag'])
bads_flag = pd.DataFrame([0]*len(df_bads_features),columns=['Flag'])

# Adding 'v' to all verbs column names
# Because they creat a conflict when joining verbs and words columns in same DataFrame
df_bads_verbs.columns = ['v'+verb for verb in df_bads_verbs.columns]
df_goods_verbs.columns = ['v'+verb for verb in df_goods_verbs.columns]

# Joining all the data in one dataframe
df_goods_data = df_goods_features.join(df_goods_verbs).join(df_goods_words).join(goods_flag)
df_bads_data = df_bads_features.join(df_bads_verbs).join(df_bads_words).join(bads_flag)
df_data = df_goods_data.append(df_bads_data)

# Separating the entry data, from the traget column
df_X = df_data.drop(columns=['Flag'])
df_Y = df_data.Flag

# Spliting the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(df_X,df_Y, test_size=0.20, random_state=42)

#normalizing the features values

#Select numerical columns which needs to be normalized
train_norm = X_train[X_train.columns[0:10]]
test_norm = X_test[X_test.columns[0:10]]

# Normalize Training Data 
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)

#Converting numpy array to dataframe
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
X_train.update(training_norm_col)

# Normalize Testing Data by using mean and SD of training set
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
X_test.update(testing_norm_col)

#create model
model = tf.keras.models.Sequential()

#get number of columns in training data
n_cols = X_train.shape[1]

#add layers to model
model.add(tf.keras.layers.Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(tf.keras.layers.Dense(250, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#train model
model.fit(X_train,y_train, epochs=5, validation_split=0.2)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")