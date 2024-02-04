import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load

# Function to apply FFT on a window of data points
def apply_fft(data_window):
    fft_result = np.fft.fft(data_window)
    return np.abs(fft_result)

# Specify the file path to your CSV file for testing
test_file_path = 'colorRandom.csv'  # Change this to your test CSV file path

# Sliding window size (±5 data points)
window_size = 11

# Load the saved regressor model
loaded_regressor = load('colormodel.joblib')

# Initialize variables to calculate regression metrics
y_true = []
y_pred = []

# Iterate through the test CSV file in chunks
for chunk_df in pd.read_csv(test_file_path, chunksize=1000, header=None):
    # Extract the target (number) for the chunk
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

    # Predict using the loaded regressor model
    y_pred_chunk = loaded_regressor.predict(X_fft_chunk)

    # Collect true and predicted values for metrics calculation
    y_true.extend(y_chunk)
    y_pred.extend(y_pred_chunk)

# Calculate regression metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
