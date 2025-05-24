import numpy as np
from tqdm.auto import trange


def kmpp(points: np.ndarray, k: int, seed: int = None, verbose: bool = False) -> np.ndarray:
    """
    Selects cluster centers using k-means++ initialization method.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        k (int): number of clusters
        seed (int, optional): random seed for reproducibility
        verbose (bool, optional): if True, show progress bar
            Default is False.
    
    Returns:
        np.ndarray: selected initial centers; shape (k, n_features)

    Notes:
        This implementation runs in O(n*k*d) time, where n is the number of samples,
        k is the number of clusters, and d is the number of features.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = points.shape
    centers = np.empty((k, n_features))
    min_squared_distances = np.full((n_samples,), np.inf)

    # Step 1: Randomly choose the first center
    centers[0] = points[np.random.randint(n_samples)]

    iterator = trange(1, k) if verbose else range(1, k)
    for i in iterator:
        # Step 2: Compute squared distances to new center
        distance_per_axis = (  # (n_samples, n_features)
            points - centers[i - 1]
        )
        squared_distance = np.linalg.norm(distance_per_axis, axis=1) ** 2  # (n_samples,)
        min_squared_distances = np.minimum(min_squared_distances, squared_distance) # (n_samples,)
        
        # Step 3: Choose next center with probability proportional to the squared distances
        probabilities = min_squared_distances / np.sum(min_squared_distances)
        next_center_index = np.random.choice(n_samples, p=probabilities)

        centers[i] = points[next_center_index]

    return centers


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from layer_flow.clustering.draw import plot_cluster_hulls

    seed = 0
    X, y = make_blobs(n_samples=1000, n_features=2, centers=6, random_state=seed)
    
    k = 6
    centroids = kmpp(X, k, seed)
    
    plot_cluster_hulls(X, centroids, y)