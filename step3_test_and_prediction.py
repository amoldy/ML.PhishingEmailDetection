import csv
import os, sys, email
import numpy as np 
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import model_from_json

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

# Compile the loaded model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#testing the model with test data
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# To make a prediction
prediction_vct = model.predict(X_test.iloc[[231]])

# If the following prints True, then the prediction was right
print(( int(y_test.iloc[231]) < 0.5 ) == ( prediction_vct[0][0] < 0.5 ))