import numpy as np


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


if __name__ == "__main__":
    # Example usage
    import time

    from layer_flow.clustering import kmpp
    from layer_flow.clustering.draw import plot_cluster_costs
    from layer_flow.data import DatasetFactory

    ds = DatasetFactory.create("blobs", num_blobs=4)
    points = ds.X

    centers = kmpp(points, k=4, seed=1)

    start_time = time.time()
    cost_per_point = compute_cost_per_point(points, centers)
    end_time = time.time()
    calc_time = end_time - start_time

    print("Time taken (compute_cost_per_point):", calc_time)

    plot_cluster_costs(points, centers, cost_per_point)
