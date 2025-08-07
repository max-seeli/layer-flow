import numpy as np
from scipy.stats import entropy


def compute_cost_per_point(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute the cost of each point to the nearest center in a memory-efficient way.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        centers (np.ndarray): centers of the clusters; shape (k, n_features)

    Returns:
        np.ndarray: cost of each point to the nearest center; shape (n_samples,)
    """
    n_samples = points.shape[0]
    costs = np.empty(n_samples, dtype=np.float64)

    costs = [np.min(np.sum((centers - point) ** 2, axis=1)) for point in points]
    return costs


def compute_clustering_cost(points: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute the total clustering cost.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        centers (np.ndarray): centers of the clusters; shape (k, n_features)

    Returns:
        float: total clustering cost
    """
    costs = compute_cost_per_point(points, centers)
    return np.sum(costs)


def compute_entropy_per_cluster(points: np.ndarray, labels: np.ndarray, centers: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute the entropy of each cluster, defined by the proximity the centers, based on the labels of the points.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        labels (np.ndarray): labels of the points; shape (n_samples,)
        centers (np.ndarray): centers of the clusters; shape (k, n_features)

    Returns:
        np.ndarray: entropy for each center; shape (k,)
    """
    n_clusters = centers.shape[0]
    
    entropies = np.zeros(n_clusters, dtype=np.float64)

    # Finding closest pairs like this is more efficient than broadcasting points via np.newaxis
    closest_centers = np.argmin([np.sum((centers - point) ** 2, axis=1) for point in points], axis=1) 

    for i in range(n_clusters):
        cluster_points = points[closest_centers == i]
        cluster_labels = labels[closest_centers == i]

        if len(cluster_points) == 0:
            entropies[i] = 0.0
            continue

        label_counts = np.bincount(cluster_labels.astype(int), minlength=np.max(labels) + 1)
        probabilities = label_counts / np.sum(label_counts)
        entropies[i] = entropy(probabilities)

    if normalize:

        max_entropy = np.log(len(np.unique(labels)))
        entropies /= max_entropy
    return entropies

def compute_mean_uniformity(points: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute the mean uniformity of clusters based on the proximity to the centers.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        labels (np.ndarray): labels of the points; shape (n_samples,)
        centers (np.ndarray): centers of the clusters; shape (k, n_features)

    Returns:
        float: mean uniformity of clusters
    """
    entropies = 1 - compute_entropy_per_cluster(points, labels, centers, normalize=True)
    return np.mean(entropies)

if __name__ == "__main__":
    # Example usage
    import time

    from layer_flow.clustering import kmpp
    from layer_flow.clustering.draw import plot_cluster_costs, plot_cluster_hulls
    from layer_flow.data import DatasetFactory

    ds = DatasetFactory.create("mnist", one_hot=False)#, num_blobs=4, one_hot=False)
    points = ds.X
    labels = ds.y

    centers = kmpp(points, k=4, seed=1)
    print("Centers:", centers)

    start_time = time.time()
    mean_uniformity = compute_mean_uniformity(points, labels, centers)
    end_time = time.time()
    calc_time = end_time - start_time    

    print("Time taken (compute_mean_uniformity):", calc_time)

    print("Mean uniformity:", mean_uniformity)

    """
    start_time = time.time()
    cost_per_point = compute_cost_per_point(points, centers)
    end_time = time.time()
    calc_time = end_time - start_time

    print("Time taken (compute_cost_per_point):", calc_time)

    plot_cluster_costs(points, centers, cost_per_point)
    plot_cluster_hulls(points, centers, labels)
    """
