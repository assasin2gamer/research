import pandas as pd
import random

# Specify the input and output file paths
input_file = 'color.csv'  # Replace with your input CSV file path
output_file = 'colorRandom.csv'  # Replace with the desired output CSV file path

# Specify the chunk size
chunk_size = 11

# Read the CSV file in chunks
chunk_list = []
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    chunk_list.append(chunk)

# Shuffle the rows within each chunk
for i, chunk in enumerate(chunk_list):
    chunk_list[i] = chunk.sample(frac=1).reset_index(drop=True)

# Concatenate the shuffled chunks and save to the output CSV file
shuffled_df = pd.concat(chunk_list, ignore_index=True)
shuffled_df.to_csv(output_file, index=False)

print(f'CSV file "{input_file}" has been randomized in chunks of {chunk_size} and saved to "{output_file}".')

