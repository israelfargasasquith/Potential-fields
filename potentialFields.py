import numpy as np
import matplotlib.pyplot as plt
from plot_with_background import plot_with_background

# Function to interactively set the goal and obstacles
def set_goal_and_obstacles(image_width, image_height, background_image):
    # Plot the background image for setting the goal
    print("Click to set the goal (1 point).")
    plt.figure()
    plt.imshow(background_image, extent=[0, image_width, 0, image_height], origin='upper')
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.title("Click to set the goal (1 point)")
    goal = plt.ginput(1, timeout=-1)  # Wait for 1 click
    plt.close()

    # Plot the background image for setting obstacles
    print("Click to set obstacles (multiple points, right-click or press Enter to finish).")
    plt.figure()
    plt.imshow(background_image, extent=[0, image_width, 0, image_height], origin='upper')
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.title("Click to set obstacles (right-click or press Enter to finish)")
    obstacles = plt.ginput(-1, timeout=-1)  # Wait for multiple clicks
    plt.close()

    return np.array(goal[0]), [np.array(obstacle) for obstacle in obstacles]

# Set the goal and obstacles interactively
image_width, image_height = 10, 10  # Example dimensions, replace with actual image dimensions
background_image = plt.imread('./Photo.png')  # Load the background image
goal, obstacles = set_goal_and_obstacles(image_width, image_height, background_image)

# Adjusted parameters
attractive_strength = 3.0
repulsive_strength = 3.0  # Reduced to balance with attractive force
repulsive_radius = 1.0     # Increased to create a smoother repulsive field
epsilon = 1e-6             # Small value to prevent division by zero

# Define the grid (match the background image dimensions)
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
            force = attractive_strength * diff / distance  # Normalize the direction
            U[i, j] = force[0]  # Directly set the attractive force
            V[i, j] = force[1]

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
U = np.divide(U, magnitude, out=np.zeros_like(U), where=magnitude > epsilon)
V = np.divide(V, magnitude, out=np.zeros_like(V), where=magnitude > epsilon)

# Set the background image path
background_path = './Photo.png'

# Call the function to plot with the background
plot_with_background(background_path, U, V, goal, obstacles)
