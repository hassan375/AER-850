import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1
# ----------------------------------------------------------------------------
# This step I loaded the provided CSV file into a dataframe
data = pd.read_csv('Project_1_Data.csv')

# Display the first few rows of the dataframe
print(data.head())


#Step 2 ----------------------------------------------------------------------
# Ensure there are no missing values and the columns are correctly formatted
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces from column names
data['Step'] = pd.to_numeric(data['Step'], errors='coerce')  # Convert Step to numeric if needed
data = data.dropna(subset=['X', 'Y', 'Z', 'Step'])  # Remove rows with missing values in key columns

# Creating the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D scatter plot
sc = ax.scatter(data['X'], data['Y'], data['Z'], c=data['Step'], cmap='viridis')

# Adding labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Scatter Plot of Coordinates Colored by Step')

# Adding color bar to represent the steps
plt.colorbar(sc, ax=ax, label='Step')

# Show the plot
plt.show()