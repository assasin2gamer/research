import matplotlib.pyplot as plt

# Read numbers from the text file into a list
with open('average2.txt', 'r') as file:
    numbers = [float(line.strip()) for line in file]

# Create a line graph
plt.plot(numbers)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph of Numbers from Text File')

# Show the graph
plt.show()