from __future__ import annotations
from typing import List, Tuple, Union, Dict

import jax
from jax import grad, jit, nn
from functools import partial
from jax import numpy as jnp
from tqdm import trange

# Uncomment the following line to enable JAX debugging for NaN values
# jax.config.update("jax_debug_nans", True)


EPSILON = 1e-7
Parameters = List[Tuple[jnp.ndarray, jnp.ndarray]]

@partial(jit, static_argnums=2)
def _activation_at_layer(
    params: Parameters, x: jnp.ndarray, layer: int
) -> jnp.ndarray:
    """
    Computes the activation at a specific layer.

    Args:
        params (Parameters): List of tuples containing weights and biases for each layer.
        x (jnp.ndarray): Input data of shape (batch_size, in_dim).
        layer (int): Layer index (0-indexed) for which to compute the activation. **Must be between 0 and len(params) - 1.**

    Returns:
        jnp.ndarray: Activation values at the specified layer.
    """
    for i in range(layer + 1):
        W, b = params[i]
        x = nn.tanh(jnp.dot(x, W) + b)
    return x

@jit
def _forward(params: Parameters, x: jnp.ndarray) -> jnp.ndarray:
    """
    Feeds a batch of inputs through the network.

    Args:
        params (Parameters): List of tuples containing weights and biases for each layer.
        x (jnp.ndarray): Input data of shape (batch_size, in_dim).

    Returns:
        jnp.ndarray: Output of the network after applying softmax activation of shape (batch_size, out_dim).
    """
    x = _activation_at_layer(params, x, len(params) - 2)
    W, b = params[-1]
    return nn.softmax(jnp.dot(x, W) + b)


@jit
def _cross_entropy(params: Parameters, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross-entropy loss between the predicted and true labels.

    Args:
        params (Parameters): List of tuples containing weights and biases for each layer.
        x (jnp.ndarray): Input data of shape (batch_size, in_dim).
        y (jnp.ndarray): True labels of shape (batch_size, out_dim).

    Returns:
        jnp.ndarray: Scalar representing the cross-entropy loss over the batch.
    """
    y_pred = _forward(params, x)
    y_pred = jnp.clip(y_pred, EPSILON, 1 - EPSILON)
    return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))


_grad_ce = jit(grad(_cross_entropy))


@jit
def _sgd_step(
    params: Parameters, x: jnp.ndarray, y: jnp.ndarray, lr: float
) -> Parameters:
    """
    Performs a single step of stochastic gradient descent on the network parameters.

    Args:
        params (Parameters): List of tuples containing weights and biases for each layer.
        x (jnp.ndarray): Input data of shape (batch_size, in_dim).
        y (jnp.ndarray): True labels of shape (batch_size, out_dim).
        lr (float): Learning rate for the update.

    Returns:
        Parameters: Updated parameters after applying the gradient descent step.
    """
    grads = _grad_ce(params, x, y)
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]


class Network:
    """
    Simple fully connected neural network.

    The network uses `tanh` activations on hidden layers and `softmax` on
    the output layer. Parameters are stored as a list of `(weights, bias)`
    tuples. All heavy numerical work is JIT-compiled via JAX for speed.
    """

    def __init__(self, parameters: Parameters = None):
        """Creates a new :class:`Network` instance.

        Args:
            parameters (Parameters, optional): Initial parameters for the network. If not provided, the network can be
                initialized later using :py:meth:`initialize` method.
        """
        self.parameters = parameters
        self._activation_at_layer = _activation_at_layer
        self._forward = _forward
        self._cross_entropy = _cross_entropy
        self._sgd_step = _sgd_step

    @classmethod
    def from_layer_sizes(cls, layer_sizes: List[int], seed: int = 0) -> Network:
        """
        Convenience constructor for creating a new :class:`Network` instance with
        specified layer sizes.

        Args:
            layer_sizes (List[int]): List of integers such that `layer_sizes[i]` is the number of units in layer `i`.
                The list length therefore equals `n_layers + 1`.
            seed (int, optional): Random seed for initializing the network parameters. Defaults to 0.

        Returns:
            Network: A new instance of the :class:`Network` class with initialized parameters.
        """
        key = jax.random.key(seed)
        subkeys = jax.random.split(key, len(layer_sizes) * 2).reshape((len(layer_sizes), 2))

        parameters = [
            (jax.random.normal(k1, (i, o)), jax.random.normal(k2, (o,)))
            for i, o, (k1, k2) in zip(layer_sizes[:-1], layer_sizes[1:], subkeys)
        ]
        return cls(parameters)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Feeds a batch of inputs through the network.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).

        Returns:
            jnp.ndarray: Output of the network after applying softmax activation of shape (batch_size, out_dim).
        """
        return self._forward(self.parameters, x)
      
    def update(self, x: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01) -> None:
        """
        Performs a single step of stochastic gradient descent on the network parameters.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            y (jnp.ndarray): True labels of shape (batch_size, out_dim).
            lr (float, optional): Learning rate for the update. Defaults to 0.01.
        """
        self.parameters = self._sgd_step(self.parameters, x, y, lr)

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the cross-entropy loss between the predicted and true labels.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            y (jnp.ndarray): True labels of shape (batch_size, out_dim).

        Returns:
            jnp.ndarray: Scalar representing the cross-entropy loss over the batch.
        """
        return self._cross_entropy(self.parameters, x, y)

    def predict(
        self, x: jnp.ndarray, labels: List[str] = None
    ) -> Union[jnp.ndarray, List[str]]:
        """
        Predicts the class labels for the input data.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            labels (List[str], optional): List of class labels. If provided, the predicted indices will be mapped to
                these labels. Defaults to None.

        Raises:
            ValueError: If the length of `labels` does not match the output dimension of the network.

        Returns:
            Union[jnp.ndarray, List[str]]: Predicted class labels. If `labels` is provided, returns a list of class labels.
                Otherwise, returns a 1D array of predicted indices.
        """
        if labels is not None and len(labels) != self.out_dim:
            raise ValueError(
                f"Labels length {len(labels)} does not match output dimension {self.out_dim}."
            )

        y_pred = self(x)
        if labels is not None:
            y_pred = jnp.argmax(y_pred, axis=1)
            return [labels[i] for i in y_pred]
        else:
            return jnp.argmax(y_pred, axis=1)
        
    def pre_activation_at_layer(
        self, x: jnp.ndarray, layer: int
    ) -> jnp.ndarray:
        """
        Computes the pre-activation values at a specific layer.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            layer (int): Layer index (0-indexed) for which to compute the pre-activation values.

        Returns:
            jnp.ndarray: Pre-activation values at the specified layer.

        Raises:
            ValueError: If the layer index is out of bounds.
        """
        if layer < 0 or layer >= len(self.parameters):
            raise ValueError(f"Layer index {layer} out of bounds.")
        
        if layer > 0:
            x = self._activation_at_layer(self.parameters, x, layer - 1)
        w, b = self.parameters[layer]
        return jnp.dot(x, w) + b
    
    def activation_at_layer(
        self, x: jnp.ndarray, layer: int
    ) -> jnp.ndarray:
        """
        Computes the activation values at a specific layer.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            layer (int): Layer index (0-indexed) for which to compute the activation values.

        Returns:
            jnp.ndarray: Activation values at the specified layer.

        Raises:
            ValueError: If the layer index is out of bounds.
        """
        if layer < 0 or layer >= len(self.parameters):
            raise ValueError(f"Layer index {layer} out of bounds.")
        
        if layer == len(self.parameters) - 1:
            return self._forward(self.parameters, x)
        return self._activation_at_layer(self.parameters, x, layer)

    @property
    def parameters(self) -> Parameters:
        """
        Current parameters of the network.

        Raises:
            ValueError: If the parameters have not yet been initialized.
        """
        if self._parameters is None:
            raise ValueError("Parameters have not been initialized.")
        return self._parameters

    @parameters.setter
    def parameters(self, params: Parameters) -> None:
        """
        Setter for fully replacing the parameters of the network.
        """
        self._parameters = params

    @property
    def n_layers(self) -> int:
        """
        Number of layers in the network.
        """
        return len(self.parameters)

    @property
    def dims(self) -> List[int]:
        """
        Layer dimensionalities inferred from the parameters shapes.
        """
        if not hasattr(self, "_dims"):
            self._dims = [w.shape[0] for w, _ in self.parameters] + [
                self.parameters[-1][0].shape[1]
            ]
        return self._dims

    @property
    def in_dim(self) -> int:
        """
        Dimensionality of the input layer.
        """
        return self.dims[0]

    @property
    def out_dim(self) -> int:
        """
        Dimensionality of the output layer.
        """
        return self.dims[-1]
