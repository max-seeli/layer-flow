from typing import Tuple, Union

import numpy as np
from tqdm import trange


def kmpp(
    points: np.ndarray,
    k: int,
    seed: int = None,
    verbose: bool = False,
    index: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Selects cluster centers using k-means++ initialization method.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        k (int): number of clusters
        seed (int, optional): random seed for reproducibility
        verbose (bool, optional): if True, show progress bar
            Default is False.
        index (bool, optional): if True, return indices of selected centers
            Default is False.

    Returns:
        np.ndarray: selected initial centers; shape (k, n_features)
        or
        Tuple[np.ndarray, np.ndarray]: tuple containing the selected initial centers and their indices
            - centers (np.ndarray): selected initial centers; shape (k, n_features)
            - indices (np.ndarray): indices of the selected centers; shape (k, )

    Notes:
        This implementation runs in O(n*k*d) time, where n is the number of samples,
        k is the number of clusters, and d is the number of features.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, _ = points.shape
    centers_idx = np.empty((k,), dtype=int)
    min_squared_distances = np.full((n_samples,), np.inf)

    # Step 1: Randomly choose the first center
    centers_idx[0] = np.random.randint(n_samples)

    iterator = trange(1, k) if verbose else range(1, k)
    for i in iterator:
        # Step 2: Compute squared distances to new center
        distance_per_axis = (  # (n_samples, n_features)
            points - points[centers_idx[i - 1]]
        )
        squared_distance = (
            np.linalg.norm(distance_per_axis, axis=1) ** 2
        )  # (n_samples,)
        min_squared_distances = np.minimum(
            min_squared_distances, squared_distance
        )  # (n_samples,)

        # Step 3: Choose next center with probability proportional to the squared distances
        probabilities = min_squared_distances / np.sum(min_squared_distances)
        next_center_index = np.random.choice(n_samples, p=probabilities)

        centers_idx[i] = next_center_index

    centers = points[centers_idx]
    if index:
        return centers, centers_idx
    else:
        return centers


if __name__ == "__main__":
    from layer_flow.clustering.draw import plot_cluster_hulls
    from layer_flow.data import DatasetFactory

    seed = 0
    k = 6
    ds = DatasetFactory.create("blobs", num_blobs=k, one_hot=False, random_state=seed)

    X, y = ds.X, ds.y
    centroids = kmpp(X, k, seed)

    plot_cluster_hulls(X, centroids, y)
