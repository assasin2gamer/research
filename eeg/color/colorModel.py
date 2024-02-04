import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import random
from joblib import dump


# Create a generator function to read the CSV in chunks
def read_csv_in_chunks(file_path, chunk_size=1000):
    reader = pd.read_csv(file_path, chunksize=chunk_size, header=None)
    for chunk in reader:
        yield chunk

# Function to apply FFT on a window of data points
def apply_fft(data_window):
    fft_result = np.fft.fft(data_window)
    return np.abs(fft_result)

# Initialize the Random Forest classifier
rf_classifier = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)

# Initialize variables to keep track of correct predictions and total predictions
correct_predictions = 0
total_predictions = 0

# Specify the file path to your CSV file
file_path = 'colorRandom.csv'

# Sliding window size (±5 data points)
window_size = 100

# Iterate through the CSV file in chunks
for chunk_df in read_csv_in_chunks(file_path, chunk_size=1000):
    # Extract the features (data columns) and target (number) for the chunk
    y_chunk = chunk_df.iloc[:, 0]  # The first column is the target variable
    X_chunk = chunk_df.iloc[:, 1:]  # Exclude the first column as it's the target label

    # Apply FFT to each channel with a sliding window
    X_fft_chunk = []
    for _, row in X_chunk.iterrows():
        row_fft = []
        for i in range(len(row)):
            if i - window_size // 2 < 0:
                window_data = row[0:i + window_size // 2 + 1].values
            elif i + window_size // 2 >= len(row):
                window_data = row[i - window_size // 2:].values
            else:
                window_data = row[i - window_size // 2:i + window_size // 2 + 1].values

            fft_result = apply_fft(window_data)
            row_fft.extend(fft_result)

        X_fft_chunk.append(row_fft)

    X_fft_chunk = pd.DataFrame(X_fft_chunk)

    # Train the classifier on the chunk with FFT features
    rf_classifier.fit(X_fft_chunk, y_chunk)



dump(rf_classifier, "colormodel.joblib")
