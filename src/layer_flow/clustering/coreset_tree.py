"""
This module implements the Coreset Tree algorithm for clustering.
"""
from __future__ import annotations
import numpy as np
from tqdm.auto import trange
from typing import Tuple

class CoresetNode:
    """
    A node in the Coreset Tree.
    """
    def __init__(self, points: np.ndarray, rep: np.ndarray):
        """
        Initialize a CoresetNodeNew instance.

        Args:
            points (np.ndarray): points of the node; shape (n_samples, n_features)
            rep (np.ndarray): representative point of the node; shape (n_features, )
        """
        self._points = points
        self.rep = rep
        self.weight = np.sum(np.linalg.norm(self._points - self.rep, axis=1) ** 2)
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
            return np.concatenate([self.left_child.points, self.right_child.points], axis=0)

    def choose_child(self) -> CoresetNode:
        """
        Choose a child node based on the weight of the points.
        """
        if self.is_leaf: # base case
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
        """
        costs = np.linalg.norm(self.points - self.rep, axis=1) ** 2
        probabilities = costs / np.sum(costs) 

        chosen_idx = np.random.choice(len(self.points), p=probabilities)
        chosen_point = self.points[chosen_idx]
        return chosen_point
    
    def split(self, point):
        """
        Split the node into two child nodes representing the newly chosen point and the representative point.

        The new child nodes consist of the points that are closer to the respective point.
        
        Args:
            point (np.ndarray): point to split the node with; shape (n_features, )
        """
        new_distances = np.linalg.norm(self.points - point, axis=1)
        rep_distances = np.linalg.norm(self.points - self.rep, axis=1)
        mask = new_distances < rep_distances
        child1 = CoresetNode(self.points[mask], point)
        child2 = CoresetNode(self.points[~mask], self.rep)

        child1.parent = self
        child2.parent = self
        self.left_child = child1
        self.right_child = child2
        self.is_leaf = False
        self._points = None  # Clear points to save memory (can be reconstructed by accessing child nodes)

        new_weight = child1.weight + child2.weight
        if self.parent is not None: # Propagate weight change to parent
            self.parent._update_weight(self.weight - new_weight)
        self.weight = new_weight

        
    def _update_weight(self, delta_weight):
        """
        Update the weight of the node and propagate the change to the parent.

        Args:
            delta_weight (float): change in weight
        """
        self.weight -= delta_weight
        if self.parent is not None:
            self.parent._update_weight(delta_weight)


def coreset_tree_root(points: np.ndarray, k: int, seed: int = None, verbose: bool = False) -> Tuple[CoresetNode, np.ndarray]:
    """
    Build a coreset tree for the given points.
    
    Args:
        points (np.ndarray): points of the dataset; shape (n_samples, n_features)
        k (int): number of clusters
        seed (int, optional): random seed for reproducibility
            Default is None.
        verbose (bool, optional): if True, show progress bar
            Default is False.
    
    Returns:
        Tuple[CoresetNodeNew, np.ndarray]: tuple containing the root of the coreset tree and the selected points
            - root (CoresetNodeNew): root of the coreset tree
            - centers (np.ndarray): selected points; shape (k, n_features)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Choose a random first point
    first_point = points[np.random.randint(len(points))]
    
    centers = [first_point]
    root = CoresetNode(points, first_point)

    iterator = trange(1, k) if verbose else range(1, k)
    for _ in iterator:
        
        # Choose a leaf node by recursively choosing a child with probability
        #  proportional to its weight
        leaf = root.choose_child()

        # Choose a point from the set of points represented by the leaf node
        #  with probability proportional to the cost of the point w.r.t. the
        #  representative point of the leaf node
        chosen_point = leaf.choose_point()
        centers.append(chosen_point)

        # Split the leaf node into two child nodes, one with the newly chosen
        #  point and the other with the representative point
        leaf.split(chosen_point)

    return root, np.array(centers)

def coreset_tree(points: np.ndarray, k: int, seed: int = None, verbose: bool = False) -> np.ndarray:
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
    """
    _, centers = coreset_tree_root(points, k, seed, verbose)
    return centers

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from layer_flow.clustering.draw import plot_cluster_hulls

    seed = 0
    X, y = make_blobs(n_samples=1000, n_features=2, centers=6, random_state=seed)
    
    k = 6
    centroids = coreset_tree(X, k, seed, verbose=True)
    
    plot_cluster_hulls(X, centroids, y)

