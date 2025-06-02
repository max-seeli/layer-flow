"""Plotting functions for clustering results.

This module provides functions to visualize clustering results, including
convex hulls of clusters and Voronoi diagrams. It is designed to work with
2D data and can be used to enhance the interpretability of clustering
algorithms.

Examples:
```
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from layer_flow.clustering.draw import plot_cluster_hulls


# Generate synthetic data
X, y = make_blobs(n_samples=1000, n_features=2, centers=6)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=seed)
kmeans.fit(X)
centroids = kmeans.cluster_centers_

# Plot clustering results
plot_cluster_hulls(X, centroids, y)
```
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull, Voronoi


def plot_cluster_costs(points, centers, costs, ax=None):
    """
    Plot the points of the dataset with colors indicating their cost to the nearest center.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        centers (np.ndarray): predicted cluster centers; shape (k, n_features)
        costs (np.ndarray): cost of each point to the nearest center; shape (n_samples,)
        ax (matplotlib.axes.Axes, optional): axes to plot on
            Default is None, which creates a new figure and axes.
    """
    owns_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
        owns_fig = True

    # Normalize costs for color mapping
    norm_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=norm_costs,
        cmap="viridis",
        s=10,
        alpha=0.8,
    )

    # Plot centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="None",
        marker="o",
        s=140,
        label="Centers",
        edgecolors="red",
        linewidths=2,
        alpha=0.8,
    )

    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12)

    plt.colorbar(scatter, label="Cost to Nearest Center")

    if owns_fig:
        plt.tight_layout()
        plt.show()


def plot_cluster_hulls(points, centers, labels, ax=None):
    """
    Plot the convex hulls of clusters in 2D space along with the Voronoi diagram of the predicted centers.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        centers (np.ndarray): predicted cluster centers; shape (k, n_features)
        labels (np.ndarray): cluster labels for each point; shape (n_samples,)
        ax (matplotlib.axes.Axes, optional): axes to plot on
            Default is None, which creates a new figure and axes.
    """
    owns_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
        owns_fig = True

    num_labels = len(np.unique(labels))
    label_map = {l: i for i, l in enumerate(np.unique(labels))}
    colors = viridis_colors(num_labels)

    vor = Voronoi(centers)

    # Plot points and their convex hulls
    for l in np.unique(labels):
        cluster_points = points[labels == l]

        if len(cluster_points) > 2:
            hull = ConvexHull(cluster_points)
            ax.fill(
                cluster_points[hull.vertices, 0],
                cluster_points[hull.vertices, 1],
                alpha=0.3,
                color=colors[label_map[l]],
            )

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[label_map[l]],
            s=10,
            label=f"Cluster {l}",
            alpha=0.8,
        )

    # Plot centers and their Voronoi diagram
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="None",
        marker="o",
        s=140,
        label="Centers",
        edgecolors="red",
        linewidths=2,
        alpha=0.8,
    )
    voronoi_plot_2d(vor, ax)

    ax.set_aspect("equal")
    ax.set_xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
    ax.set_ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12)

    if owns_fig:
        plt.tight_layout()
        plt.show()


def voronoi_plot_2d(vor: Voronoi, ax: plt.Axes):
    """
    Plot the given Voronoi diagram in 2-D

    Args:
        vor (scipy.spatial.Voronoi): Voronoi diagram to plot
        ax (matplotlib.axes.Axes, optional): Axes to plot on

    Note:
        This function is adapted from the `scipy.spatial.voronoi_plot_2d` function.
    """
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if vor.furthest_site:
                direction = -direction
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(
        LineCollection(
            finite_segments + infinite_segments,
            colors="k",
            lw=1,
            alpha=0.5,
            linestyle="dashed",
            zorder=-1,
            label="Center Boundaries",
        )
    )


def viridis_colors(n):
    """
    Generate a list of n colors from the viridis colormap.

    Args:
        n (int): number of colors to generate

    Returns:
        list: list of RGB tuples
    """
    cmap = get_cmap("viridis")
    return cmap(np.linspace(0, 1, n))
