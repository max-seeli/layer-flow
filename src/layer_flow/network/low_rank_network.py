from __future__ import annotations

from typing import List, Tuple
from jax import grad, jit, random, nn
import jax
import jax.numpy as jnp
from functools import partial

from layer_flow.network import Network
from layer_flow.clustering import coreset_tree, compute_clustering_cost


EPSILON = 1e-7
LowRankParameters = List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]  # (L, D, b)

@partial(jit, static_argnums=2)
def _activation_at_layer_lowrank(
    params: LowRankParameters, x: jnp.ndarray, layer: int
) -> jnp.ndarray:
    for i in range(layer + 1):
        L, D, b = params[i]
        x1 = x @ L
        x2 = x1 @ D
        x3 = x2 @ L.T
        
        #W = L @ D @ L.T
        x = nn.tanh(x3 + b)
    return x

@jit
def _forward_lowrank(params: LowRankParameters, x: jnp.ndarray) -> jnp.ndarray:
    x = _activation_at_layer_lowrank(params, x, len(params) - 2)
    L, D, b = params[-1]
    
    x1 = x @ L
    x2 = x1 @ D
    
    #W = L @ D
    return nn.softmax(x2 + b)

@jit
def _cross_entropy_lowrank(params: LowRankParameters, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    y_pred = _forward_lowrank(params, x)
    y_pred = jnp.clip(y_pred, EPSILON, 1 - EPSILON)
    return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

@jit
def _cross_entropy_lowrank_diff(diff_params: List[Tuple[jnp.ndarray, jnp.ndarray]], undiff_params: List[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    params = [(L, D, b) for ((D, b), L) in zip(diff_params, undiff_params)]
    y_pred = _forward_lowrank(params, x)
    y_pred = jnp.clip(y_pred, EPSILON, 1 - EPSILON)
    return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))



_grad_ce_lowrank = jit(grad(_cross_entropy_lowrank_diff))

@jit
def _sgd_step_lowrank(params: LowRankParameters, x: jnp.ndarray, y: jnp.ndarray, lr: float) -> LowRankParameters:
    diff_param = [(D, b) for (L, D, b) in params]
    undiff_param = [L for (L, _, _) in params]
    grads = _grad_ce_lowrank(diff_param, undiff_param, x, y)
    # Only D and b are trained
    return [(L, D - lr * gD, b - lr * gb) for (L, D, b), (gD, gb) in zip(params, grads)]

class LowRankNetwork(Network):
    def __init__(self, parameters: LowRankParameters = None):
        """
        Initializes a LowRankNetwork with the given parameters.

        Args:
            parameters (LowRankParameters, optional): A list of tuples containing the low-rank parameters (L, D, b).
                If None, an empty network is created.
        """
        super().__init__(parameters)

        self._activation_at_layer = _activation_at_layer_lowrank
        self._forward = _forward_lowrank
        self._cross_entropy = _cross_entropy_lowrank
        self._sgd_step = _sgd_step_lowrank

    @classmethod
    def from_layer_sizes(cls, layer_sizes: List[int], seed: int = 0) -> LowRankNetwork:
        raise NotImplementedError(
            "LowRankNetwork.from_layer_sizes is not implemented. Use LowRankNetwork.from_depth_rank instead."
        )
        
    @classmethod
    def from_depth_rank(
        cls,
        depth: int,
        rank: int,
        in_dim: int,
        out_dim: int,
        seed: int = 0,
    ) -> LowRankNetwork:
        """
        Creates a LowRankNetwork with the specified depth and rank.

        Args:
            depth (int): The number of layers in the network.
            rank (int): The rank of the low-rank decomposition.
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.

        Returns:
            LowRankNetwork: A new instance of LowRankNetwork.
        """
        key = random.PRNGKey(seed)
        subkeys = jax.random.split(key, depth * 3).reshape((depth, 3, -1))

        params = []
        for k1, k2, k3 in subkeys[:-1]:
            L = random.normal(k3, (in_dim, rank))
            D = random.normal(k1, (rank, rank))
            b = random.normal(k2, (in_dim,))
            params.append((L, D, b))
        
        # Last layer
        k1, k2, k3 = subkeys[-1]
        L = random.normal(k3, (in_dim, out_dim))
        D = random.normal(k1, (out_dim, out_dim))
        b = random.normal(k2, (out_dim,))
        params.append((L, D, b))
        return cls(params)
    

    @classmethod
    def from_coreset_depth_rank(
        cls,
        data: jnp.ndarray,
        depth: int,
        rank: int,
        in_dim: int,
        out_dim: int,
        seed: int = 0,
    ) -> LowRankNetwork:
        """
        Creates a LowRankNetwork with the specified depth and rank, using a coreset for initialization.

        Args:
            data (jnp.ndarray): The input data to create the coreset.
            depth (int): The number of layers in the network.
            rank (int): The rank of the low-rank decomposition.
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        Returns:
            LowRankNetwork: A new instance of LowRankNetwork initialized with a coreset.
        """
        key = random.PRNGKey(seed)
        subkeys = jax.random.split(key, depth * 3).reshape((depth, 3, -1))

        params = []
        print(data.shape)
        coreset = coreset_tree(data, rank, seed=seed, verbose=True)
        print("Cost:", compute_clustering_cost(data, coreset))
        print("Coreset rank:", jnp.linalg.matrix_rank(coreset))
        
        for k1, k2, k3 in subkeys[:-1]:
            L = coreset.T
            # Normalize L
            L = (L - jnp.mean(L)) / (jnp.std(L) + EPSILON)
            print("L1:", L)
            D = random.normal(k1, (rank, rank))
            b = random.normal(k2, (in_dim,))

            X_prime = data @ L @ D @ L.T + b
            coreset = coreset_tree(X_prime, rank, seed=seed, verbose=True)

            params.append((L, D, b))
        
        # Last layer
        #k1, k2 = subkeys[-1]
        L = coreset.T
        # Normalize L
        L = (L - jnp.mean(L)) / (jnp.std(L) + EPSILON)
        print("L2:", L)
        print("L2 rank:", jnp.linalg.matrix_rank(L))

        k1, k2, k3 = subkeys[-1]
        L = random.normal(k3, (in_dim, rank))
        D = random.normal(k1, (rank, out_dim))
        b = random.normal(k2, (out_dim,))
        params.append((L, D, b))

        return cls(params)
    

