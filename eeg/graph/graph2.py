import matplotlib.pyplot as plt

# Read numbers from the text file into a list
with open('average2.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists to store the averages and the current sum
averages = []
current_sum = 0

# Iterate through the lines and calculate averages for every 12 lines
for i, line in enumerate(lines):
    value = float(line.strip())
    current_sum += value
    
    if (i + 1) % 12 == 0:
        # Calculate the average for the 12 lines and append to the list
        average = current_sum / 12
        averages.append(average)
        
        # Reset the current sum for the next 12 lines
        current_sum = 0

# Create a line graph for the averages
plt.plot(averages)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph of Averages from Text File (Every 12 Lines)')

# Show the graph
plt.show()