# load losses_3000.npy from losses_3000.npy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the losses from the .npy file
losses = np.load('results/losses_40000.npy')
x = np.log10(np.arange(len(losses))+1).reshape(-1, 1)  # Reshape to 2D array
y = np.log10(losses).reshape(-1, 1)  # Reshape to 2D array    
model = LinearRegression()
model.fit(x, y)

# Predict the line
y_pred = model.predict(x)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the losses
ax.plot(losses, label='Loss', color='blue')

# Plot the fitted line
slope = model.coef_[0][0]
intercept = model.intercept_[0]
ax.plot(np.power(10,x), np.power(10,y_pred), color='red', linestyle='--', label='Fitted Line', linewidth=2, zorder=10)

# Set the title and labels
ax.set_title(f'Training Loss Over Time with Fitted Line\nSlope: {slope:.4f}, Intercept: {intercept:.4f}')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Loss Value')

# Add a grid
ax.grid(True)

# Add a legend
ax.legend()

# log y-axis
ax.set_yscale('log')
# log x-axis
ax.set_xscale('log')

plt.show()