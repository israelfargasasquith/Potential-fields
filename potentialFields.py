import numpy as np
import matplotlib.pyplot as plt
from plot_with_background import plot_with_background

# Define parameters
goal = np.array([5, 5])
# Define obstacles to create a "corridor"
obstacles = [
# np.array([3, 3]), np.array([3, 7]),
#  np.array([7, 3]), np.array([7, 7]),
#   np.array([1, 7]), np.array([1, 3]),
#   np.array([9, 7]), np.array([9, 3]),
#    np.array([5, 7]), np.array([5, 3])
]
# Adjusted parameters
attractive_strength = 1.0
repulsive_strength = 50.0  # Reduced to balance with attractive force
repulsive_radius = 1.5     # Increased to create a smoother repulsive field
epsilon = 1e-6             # Small value to prevent division by zero

# Define the grid
x_range = np.linspace(0, 10, 20)
y_range = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x_range, y_range)

# Initialize potential field components
U = np.zeros(X.shape)
V = np.zeros(Y.shape)

# Compute the attractive potential
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos = np.array([X[i, j], Y[i, j]])
        diff = goal - pos
        distance = np.linalg.norm(diff)
        if distance > epsilon:  # Ensure distance is greater than epsilon
            force = attractive_strength * diff / distance
            U[i, j] += force[0]
            V[i, j] += force[1]

# Compute the repulsive potential
for obstacle in obstacles:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j]])
            diff = pos - obstacle
            distance = max(np.linalg.norm(diff), epsilon)  # Clamp distance to at least epsilon
            if distance < repulsive_radius:
                force = repulsive_strength * (1.0 / distance - 1.0 / repulsive_radius) * (diff / (distance**3))
                U[i, j] += force[0]
                V[i, j] += force[1]

# Normalize the vectors to improve visualization
magnitude = np.sqrt(U**2 + V**2)
U = np.divide(U, magnitude, out=np.zeros_like(U), where=magnitude > 0)
V = np.divide(V, magnitude, out=np.zeros_like(V), where=magnitude > 0)

# Set the background image path
background_path = './Photo.png'

# Call the function to plot with the background
plot_with_background(background_path, U, V, goal, obstacles)
