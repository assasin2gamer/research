import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('color.csv')

# Extract the second column (assuming it's labeled as "Column2")
y = df.iloc[:, 1]  # Change the column index if necessary

# Create x-axis values as row count
x = range(1, len(y) + 1)

# Plot the data
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.plot(x, y)
plt.title('Plot of Column 2')
plt.xlabel('Row Count')
plt.ylabel('Column 2')
plt.grid(True)
plt.show()
