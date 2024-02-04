import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.fft import fft
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import math

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import re
import joblib

window_size = 2
test_Y = []
test_X = []

def test():

    csv_reader = pd.read_csv("./test2.csv", chunksize=10)
    Y_chunk = []
    X_chunk = []
    for chunk in csv_reader:
        for index, row in chunk.iterrows():
            coordinates_match = re.match(r"\((.*),(.*)\)", row.iloc[0])
            if coordinates_match:
                x = float(coordinates_match.group(1))
                y = float(coordinates_match.group(2))
                coordinates_array = [x, y]
                Y_chunk.append(coordinates_array)
            X_chunk.append(row.iloc[1:].values.tolist())
        processing(X_chunk, Y_chunk)
        X_chunk = []
        Y_chunk = []

def processing(X_chunk, Y_chunk):
    global window_size

    processed_X = []

    for i in range(len(X_chunk) - 5 + 1):
        if i > 5:
            X_window = X_chunk[i - window_size:i + window_size]
            for channels in X_window:
                channels = np.abs(fft(channels))
            flat_data = (np.array(X_window).flatten().tolist())
            if(len(flat_data) == 56):
                processed_X.append(flat_data)
  
        else:
            X_window = X_chunk[i:i + window_size + window_size]
            for channels in X_window:
                channels = np.abs(fft(channels))
            flat_data = (np.array(X_window).flatten().tolist())
            if(len(flat_data) == 56):
                processed_X.append(flat_data)
        #print(len(flat_data))

    
    model = joblib.load('my_random_forest_model.pkl')
    prediction = model.predict(processed_X)
    #print("prediction :" + str(prediction) + " Actual: " + str(Y_chunk))
    
    
    distance_all = []
    
    for count in range(len(prediction)):
        dx = prediction[count][0] - Y_chunk[count][0]
        dy = prediction[count][1] - Y_chunk[count][1]
        distance = math.sqrt(dx**2 + dy**2)
        distance_all.append(distance)
        
        
    array_sum = sum(distance_all)
    average = array_sum / len(distance_all)
    print("Average:", average)
    
        
    
    
test()