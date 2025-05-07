import numpy as np


def kmeans_plus_plus_init(points: np.ndarray, k: int, seed: np.ndarray = None):
    """
    Selects initial cluster centers using k-means++ initialization method.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        k (int): number of clusters
        seed (int, optional): random seed for reproducibility

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

    for i in range(1, k):
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
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    seed = 0
    X, y = make_moons(n_samples=100, noise=0.1, random_state=seed)

    centroids = kmeans_plus_plus_init(X, 4, seed)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="teal",
        label="Class 0",
        alpha=0.7,
        edgecolors="k",
        s=80,
        linewidths=1,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="goldenrod",
        label="Class 1",
        alpha=0.7,
        edgecolors="k",
        s=80,
        linewidths=1,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="None",
        marker="o",
        s=140,
        label="Centroids",
        edgecolors="red",
        linewidths=4,
    )
    plt.title("K-means++ Initialization on Moons Dataset", fontsize=24)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=True, fontsize=10, loc="upper right")

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
