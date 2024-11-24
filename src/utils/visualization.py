import matplotlib.pyplot as plt

def plot_images(images, title="Images", num_images=9):
    """
    Plots a grid of images.

    Args:
        images (numpy.ndarray): Array of images to plot.
        title (str): Title of the plot.
        num_images (int): Number of images to display in the grid.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.suptitle(title)
    plt.show()
