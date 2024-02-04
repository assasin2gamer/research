import numpy as np
import matplotlib.pyplot as plt

# Read numbers from the text file into a list
with open('average2.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists to store the original values
original_values = []

# Iterate through the lines and store the original values
for line in lines:
    value = float(line.strip())
    original_values.append(value)

# Initialize lists to store the averaged values
averages = []

# Calculate the averages for every 12 data points
current_sum = 0
for i, value in enumerate(original_values):
    current_sum += value

    if (i + 1) % 12 == 0:
        average = current_sum / 12
        averages.append(average)
        current_sum = 0

# Create an array for the x-values (indices) and y-values (averages)
x_values = np.arange(len(averages))
y_values = np.array(averages)

# Perform linear regression to calculate the line of best fit
slope, intercept = np.polyfit(x_values, y_values, 1)

# Create the line of best fit using the calculated slope and intercept
line_of_best_fit = slope * x_values + intercept

# Create a line graph for the averages
plt.plot(x_values, y_values, label='Averages')

# Plot the line of best fit
plt.plot(x_values, line_of_best_fit, label='Line of Best Fit', linestyle='--')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph of Averages with Line of Best Fit (Every 12 Data Points)')

# Calculate the R-squared value
correlation_matrix = np.corrcoef(x_values, y_values)
r_squared = correlation_matrix[0, 1]**2

# Display the R-squared value
print(f'R-squared value: {r_squared:.4f}')

# Show the graph with a legend
plt.legend()
plt.show()
