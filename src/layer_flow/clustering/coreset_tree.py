from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from tqdm import trange


class CoresetNode:
    """
    A node in the Coreset Tree.
    """

    def __init__(
        self, points: np.ndarray, rep: np.ndarray, rep_distances: np.ndarray = None
    ):
        """
        Initialize a CoresetNodeNew instance.

        Args:
            points (np.ndarray): points of the node; shape (n_samples, n_features)
            rep (np.ndarray): representative point of the node; shape (n_features, )
            rep_distances (np.ndarray, optional): distances of the points to the representative point; shape (n_samples, )
                Default is None, which will be computed from points and rep.
        """
        self._points = points
        self.rep = rep
        if rep_distances is None:
            self.rep_distances = np.linalg.norm(points - rep, axis=1)
        else:
            self.rep_distances = rep_distances
        self.weight = np.sum(self.rep_distances**2)
        self.parent: CoresetNode = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = True

    @property
    def points(self) -> np.ndarray:
        """
        Get the points of the node.

        Returns:
            np.ndarray: points of the node; shape (n_samples, n_features)
        """
        if self.is_leaf:
            return self._points
        else:
            return np.concatenate(
                [self.left_child.points, self.right_child.points], axis=0
            )

    def choose_child(self) -> CoresetNode:
        """
        Choose a child node based on the weight of the points.
        """
        if self.is_leaf:  # base case
            return self
        else:
            prob_l = self.left_child.weight / self.weight
            if np.random.rand() < prob_l:
                return self.left_child.choose_child()
            else:
                return self.right_child.choose_child()

    def choose_point(self):
        """
        Choose a point based on the squared distance of the point to the representative point.

        Returns:
            np.ndarray: chosen point; shape (n_features, )
            int: index of the chosen point in the original points array
        """
        costs = self.rep_distances**2
        probabilities = costs / np.sum(costs)

        chosen_idx = np.random.choice(len(self.points), p=probabilities)
        chosen_point = self.points[chosen_idx]
        return chosen_point, chosen_idx

    def split(self, point: np.ndarray):
        """
        Split the node into two child nodes representing the newly chosen point and the representative point.

        The new child nodes consist of the points that are closer to the respective point.

        Args:
            point (np.ndarray): point to split the node with; shape (n_features, )
        """
        new_distances = np.linalg.norm(self.points - point, axis=1)
        mask = new_distances < self.rep_distances
        child1 = CoresetNode(self.points[mask], point, new_distances[mask])
        child2 = CoresetNode(self.points[~mask], self.rep, self.rep_distances[~mask])

        child1.parent = self
        child2.parent = self
        self.left_child = child1
        self.right_child = child2
        self.is_leaf = False
        self._points = None  # Clear points to save memory (can be reconstructed by accessing child nodes)
        self.rep_distances = None  # Clear distances to save memory

        new_weight = child1.weight + child2.weight
        if self.parent is not None:  # Propagate weight change to parent
            self.parent._update_weight(self.weight - new_weight)
        self.weight = new_weight

    def _update_weight(self, delta_weight: float):
        """
        Update the weight of the node and propagate the change to the parent.

        Args:
            delta_weight (float): change in weight
        """
        self.weight -= delta_weight
        if self.parent is not None:
            self.parent._update_weight(delta_weight)


def coreset_tree_root(
    points: np.ndarray,
    k: int,
    seed: int = None,
    verbose: bool = False,
    index: bool = False,
) -> Union[Tuple[CoresetNode, np.ndarray], Tuple[CoresetNode, np.ndarray, np.ndarray]]:
    """
    Build a coreset tree for the given points.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        k (int): number of clusters
        seed (int, optional): random seed for reproducibility
            Default is None.
        verbose (bool, optional): if True, show progress bar
            Default is False.
        index (bool, optional): if True, return the indices of the selected points
            Default is False.

    Returns:
        Tuple[CoresetNodeNew, np.ndarray]: tuple containing the root of the coreset tree and the selected points
            - root (CoresetNodeNew): root of the coreset tree
            - centers (np.ndarray): selected points; shape (k, n_features)
        or
        Tuple[CoresetNodeNew, np.ndarray, np.ndarray]: tuple containing the root of the coreset tree, the selected points, and their indices
            - root (CoresetNodeNew): root of the coreset tree
            - centers (np.ndarray): selected points; shape (k, n_features)
            - indices (np.ndarray): indices of the selected points; shape (k, )
    """
    if seed is not None:
        np.random.seed(seed)

    centers_idx = np.empty((k,), dtype=int)

    # Choose a random first point
    centers_idx[0] = np.random.randint(len(points))

    root = CoresetNode(points, points[centers_idx[0]])

    iterator = trange(1, k) if verbose else range(1, k)
    for i in iterator:
        # Choose a leaf node by recursively choosing a child with probability
        #  proportional to its weight
        leaf = root.choose_child()

        # Choose a point from the set of points represented by the leaf node
        #  with probability proportional to the cost of the point w.r.t. the
        #  representative point of the leaf node
        chosen_point, chosen_idx = leaf.choose_point()
        centers_idx[i] = chosen_idx

        # Split the leaf node into two child nodes, one with the newly chosen
        #  point and the other with the representative point
        leaf.split(chosen_point)

    centers = points[centers_idx]
    if index:
        return root, centers, centers_idx
    else:
        return root, centers


def coreset_tree(
    points: np.ndarray,
    k: int,
    seed: int = None,
    verbose: bool = False,
    index: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build a coreset for the given points.

    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        k (int): number of clusters
        seed (int, optional): random seed for reproducibility
            Default is None.
        verbose (bool, optional): if True, show progress bar
            Default is False.

    Returns:
        np.ndarray: selected coreset points; shape (k, n_features)
        or
        Tuple[np.ndarray, np.ndarray]: tuple containing the selected coreset points and their indices
            - centers (np.ndarray): selected coreset points; shape (k, n_features)
            - indices (np.ndarray): indices of the selected coreset points; shape (k, )
    """
    results = coreset_tree_root(points, k, seed, verbose, index)
    if index:
        centers, indices = results[1], results[2]
        return centers, indices
    else:
        centers = results[1]
        return centers


"""
if __name__ == "__main__":
    from layer_flow.clustering.draw import plot_cluster_hulls
    from layer_flow.data import DatasetFactory

    seed = 0
    k = 6
    ds = DatasetFactory.create("blobs", num_blobs=k, one_hot=False, random_state=seed)

    X, y = ds.X, ds.y
    centroids = coreset_tree(X, k, seed)

    plot_cluster_hulls(X, centroids, y)
"""

if __name__ == "__main__":
    from layer_flow.data import DatasetFactory

    ds = DatasetFactory.create("mnist")
    X, y = ds.X, ds.y

    k = 30
    centroids = coreset_tree(X, k, 42, verbose=True)
