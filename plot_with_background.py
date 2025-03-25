import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_with_background(background_path, U, V, goal, obstacles):
    """
    Plots a potential field with a PNG background.

    Parameters:
        background_path (str): Path to the PNG image.
        U (ndarray): X components of the vector field.
        V (ndarray): Y components of the vector field.
        goal (tuple): Coordinates of the goal (x, y).
        obstacles (list): List of obstacle coordinates [(x1, y1), (x2, y2), ...].
    """
    # Load the PNG image
    background_image = mpimg.imread(background_path)

    # Get the dimensions of the image
    image_height, image_width, _ = background_image.shape

    # Plot the background image
    plt.figure(figsize=(10, 10))
    plt.imshow(background_image, extent=[0, image_width, 0, image_height], origin='upper')  # Flip Y-axis origin

    # Adjust the grid to match the image dimensions
    x_range = np.linspace(0, image_width, U.shape[1])
    y_range = np.linspace(0, image_height, U.shape[0])
    X, Y = np.meshgrid(x_range, y_range)

    # Overlay the potential field vectors
    plt.quiver(X, Y, U, V, color='blue')
    plt.scatter(goal[0], goal[1], color='red', marker='*', s=200, label='Goal')
    for obstacle in obstacles:
        plt.scatter(obstacle[0], obstacle[1], color='black', marker='o', s=100, label='Obstacle')

    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential Field with Background Image')
    plt.grid(True)
    #plt.legend()
    plt.show()
