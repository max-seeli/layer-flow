import numpy as np
from tqdm import trange


def lloyd(
    points: np.ndarray,
    centers: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = True,
) -> np.ndarray:
    """
    Lloyd's algorithm for k-means clustering.

    Args:
        points (np.ndarray): Data points of shape (n_samples, n_features).
        centers (np.ndarray): Initial cluster centers of shape (k, n_features).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        verbose (bool): Whether to print progress messages.

    Returns:
        np.ndarray: Final cluster centers after convergence.
    """
    n_samples, n_features = points.shape
    k = centers.shape[0]

    iterator = (
        trange(max_iter, desc="Lloyd's Algorithm") if verbose else range(max_iter)
    )
    for i in iterator:
        # Assign clusters
        closest_centers = np.argmin(
            [np.sum((centers - point) ** 2, axis=1) for point in points], axis=1
        )

        # Update centers
        new_centers = np.array(
            [points[closest_centers == c].mean(axis=0) for c in range(k)]
        )

        if np.linalg.norm(new_centers - centers) < tol:
            if verbose:
                print(f"Converged after {i} iterations.")
            break

        centers = new_centers

    return centers


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from layer_flow import RESULTS_DIR
    from layer_flow.clustering import kmpp
    from layer_flow.clustering.draw import plot_cluster_hulls, save_plot
    from layer_flow.data import DatasetFactory

    seed = 0
    k = 6
    ds = DatasetFactory.create("blobs", num_blobs=k, one_hot=False, random_state=seed)

    X, y = ds.X, ds.y
    initial_centers = kmpp(X, k=k)
    final_centers = lloyd(X, initial_centers)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title("Initial Centers")
    axes[1].set_title("Final Centers")
    plot_cluster_hulls(X, initial_centers, y, ax=axes[0])
    plot_cluster_hulls(X, final_centers, y, ax=axes[1])
    save_plot(fig, RESULTS_DIR / "lloyd_clustering.png")
