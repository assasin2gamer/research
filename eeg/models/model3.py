import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.fft import fft
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

from tensorflow import keras
import joblib
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import re


window_size = 100
test_Y = []
test_X = []


def data_generator(file_path, chunk_size):
    csv_reader = pd.read_csv("./pixel2.csv", chunksize=1000)
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
    processed_X = []
    for i in range(len(X_chunk) - window_size + 1):
        # Choose the appropriate window based on the index
        if i > window_size:
            X_window = X_chunk[i - window_size:i + window_size]
        else:
            X_window = X_chunk[i:i + 2 * window_size]

        # Apply FFT and calculate energy for each channel in the window
        energy_data = []
        for channel in X_window:
            fft_coeffs = np.abs(fft(channel))  # Compute the FFT
            energy = np.sum(fft_coeffs ** 2)   # Calculate energy
            energy_data.append(energy)

        flat_data = np.array(energy_data).flatten().tolist()

        # Append the processed data if it matches the expected length
        if (len(flat_data) == 200):
            processed_X.append(flat_data)
     
  

    # Train the model with the processed data
    train(processed_X, Y_chunk[:len(processed_X)])
        
 
    


def defineModel():
    global model
    model = RandomForestRegressor(n_estimators=100, random_state=100, n_jobs=-1)  # Create a Random Forest model with 100 estimators (you can adjust this number)
    

def train(X_chunk, Y_chunk):
    global test_X
    global test_Y
    print("intitiate")
    model.fit(X_chunk, Y_chunk)  # Train the Random Forest model
    print("computed")
    joblib.dump(model, "my_random_forest_model2.pkl")
    test()



def main():
    global model
    defineModel()
    data_generator("./pixel.csv", 100)



def test():

    csv_reader = pd.read_csv("./test2.csv", chunksize=1000)
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
        processing_test(X_chunk, Y_chunk)
        X_chunk = []
        Y_chunk = []

def processing_test(X_chunk, Y_chunk):
    global window_size

    processed_X = []

    for i in range(len(X_chunk) - window_size + 1):
        # Choose the appropriate window based on the index
        if i > window_size:
            X_window = X_chunk[i - window_size:i + window_size]
        else:
            X_window = X_chunk[i:i + 2 * window_size]

        # Apply FFT and calculate energy for each channel in the window
        energy_data = []
        for channel in X_window:
            fft_coeffs = np.abs(fft(channel))  # Compute the FFT
            energy = np.sum(fft_coeffs ** 2)   # Calculate energy
            energy_data.append(energy)

        flat_data = np.array(energy_data).flatten().tolist()

        # Append the processed data if it matches the expected length
     
        if len(flat_data) == 200:
            processed_X.append(flat_data)

    # Write the first elements of the original data to a file
    with open("example.txt", "a") as file:
        file.write(f"{X_chunk[0]} {Y_chunk[0]}\n")

    
    model = joblib.load('my_random_forest_model2.pkl')
    try:
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
        with open("average2.txt", "a") as file:
            file.write(str(average) + "\n")
    except:
        print("uwu")
        
        
        
        
main()
