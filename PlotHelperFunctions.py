
from matplotlib import pyplot as plt
import numpy as np

def plot_clustering(points, clusters, title):
    """
    Plots the given clustering.

    Parameters:
    - points (list or array): A list or array of points where each entry is a list/array of coordinates.
    - clusters (dict): A dictionary where keys are point indices (0 to n-1) and values are cluster indices.
    """
    # Convert points to a numpy array for easier processing
    data = np.array(points)
    cluster_labels = np.array([clusters[i] for i in range(len(points))])  # Extract cluster labels

    # Check if the points are 2D
    if data.shape[1] != 2:
        raise ValueError("Only 2D data can be visualized. Your data has {} dimensions.".format(data.shape[1]))

    # Scatter plot for clustering
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap="viridis", s=50, alpha=0.6)

    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=set(cluster_labels), title="Clusters")
    plt.title(title)
    plt.xlabel("Coordinate 1")
    plt.ylabel("Coordinate 2")
    plt.grid()
    plt.show()

def plot_stuff(points, clusters1, clusters2, clusters3, title="",title_1 = "", title_2 = "", title_3 = "",ground_truth=None):
    data = np.array(points)
    cluster_labels1 = np.array([clusters1[i] for i in range(len(points))])  # Extract cluster labels
    cluster_labels2 = np.array([clusters2[i] for i in range(len(points))])
    cluster_labels3 = np.array([clusters3[i] for i in range(len(points))])

    # Check if the points are 2D
    if data.shape[1] != 2:
        raise ValueError("Only 2D data can be visualized. Your data has {} dimensions.".format(data.shape[1]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18, 6))
    fig.suptitle(title)
    ax1.scatter(data[:, 0], data[:, 1], c=cluster_labels1, cmap="viridis", s=50, alpha=0.6)
    ax1.title.set_text(title_1)
    ax2.scatter(data[:, 0], data[:, 1], c=cluster_labels2, cmap="viridis", s=50, alpha=0.6)
    ax2.title.set_text(title_2)
    ax3.scatter(data[:, 0], data[:, 1], c=cluster_labels3, cmap="viridis", s=50, alpha=0.6)
    ax3.title.set_text(title_3)


    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()
