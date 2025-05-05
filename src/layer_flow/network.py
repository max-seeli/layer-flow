from __future__ import annotations
from typing import List, Tuple, Union, Dict

import jax
from jax import grad, jit, nn
from jax import numpy as jnp
from tqdm.auto import trange

# Uncomment the following line to enable JAX debugging for NaN values
# jax.config.update("jax_debug_nans", True)


EPSILON = 1e-7
Parameters = List[Tuple[jnp.ndarray, jnp.ndarray]]


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
    for w, b in params[:-1]:
        x = nn.tanh(jnp.dot(x, w) + b)
    w, b = params[-1]
    return nn.softmax(jnp.dot(x, w) + b)


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
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]


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
        instance = cls()
        instance.initialize(layer_sizes, seed)
        return instance

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Feeds a batch of inputs through the network.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).

        Returns:
            jnp.ndarray: Output of the network after applying softmax activation of shape (batch_size, out_dim).
        """
        return _forward(self.parameters, x)

    def initialize(self, layer_sizes: List[int], seed: int) -> None:
        """
        Initializes network parameters with standard normal distribution samples.

        Args:
            layer_sizes (List[int]): List of integers such that `layer_sizes[i]` is the number of units in layer `i`.
                The list length therefore equals `n_layers + 1`.
            seed (int): Random seed for initializing the network parameters
        """
        key = jax.random.key(seed)
        subkeys = jax.random.split(key, len(layer_sizes) * 2).reshape((len(layer_sizes), 2))

        self.parameters = [
            (jax.random.normal(k1, (i, o)), jax.random.normal(k2, (o,)))
            for i, o, (k1, k2) in zip(layer_sizes[:-1], layer_sizes[1:], subkeys)
        ]

    def train(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_val: jnp.ndarray,
        y_val: jnp.ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
        log_every: int = 100,
        verbose: bool = False,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Optimizes the network parameters using stochastic gradient descent.

        Args:
            x_train (jnp.ndarray): Training data of shape (n_samples, in_dim).
            y_train (jnp.ndarray): Training labels of shape (n_samples, out_dim).
            x_val (jnp.ndarray): Validation data of shape (n_samples, in_dim).
            y_val (jnp.ndarray): Validation labels of shape (n_samples, out_dim).
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            epochs (int, optional): Number of training epochs. Defaults to 1000.
            log_every (int, optional): Log every `log_every` epochs. Defaults to 100.
            verbose (bool, optional): If True, print training and validation loss at each log step. Defaults to False.

        Returns:
            Tuple[Dict[int, float], Dict[int, float]]: A tuple containing two dictionaries:
                - The first dictionary contains training losses at each logged epoch.
                - The second dictionary contains validation losses at each logged epoch.
        """
        losses = {}
        val_losses = {}
        for epoch in (pbar := trange(1, epochs + 1, desc="Training", unit="epoch")):
            self.update(x_train, y_train, lr)
            if epoch % log_every == 0 or epoch in [1, epochs]:
                loss = self.loss(x_train, y_train)
                val_loss = self.loss(x_val, y_val)
                if verbose:
                    print(f"@[{epoch}]: train loss {loss}; val loss {val_loss}")
                else:
                    pbar.set_postfix({"train loss": loss, "val loss": val_loss})
                losses[epoch] = loss
                val_losses[epoch] = val_loss
        return losses, val_losses

    def update(self, x: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01) -> None:
        """
        Performs a single step of stochastic gradient descent on the network parameters.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            y (jnp.ndarray): True labels of shape (batch_size, out_dim).
            lr (float, optional): Learning rate for the update. Defaults to 0.01.
        """
        self.parameters = _sgd_step(self.parameters, x, y, lr)

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the cross-entropy loss between the predicted and true labels.

        Args:
            x (jnp.ndarray): Input data of shape (batch_size, in_dim).
            y (jnp.ndarray): True labels of shape (batch_size, out_dim).

        Returns:
            jnp.ndarray: Scalar representing the cross-entropy loss over the batch.
        """
        return _cross_entropy(self.parameters, x, y)

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


def accuracy(model: Network, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """
    Computes the classification accuracy of `model` on the data.

    Args:
        model (Network): The trained model.
        x (jnp.ndarray): Input data of shape (batch_size, in_dim).
        y (jnp.ndarray): True labels of shape (batch_size, out_dim).

    Returns:
        jnp.ndarray: Classification accuracy as a float between 0 and 1.
    """
    pred = model.predict(x)
    return jnp.mean(jnp.argmax(y, axis=1) == pred)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
    y = nn.one_hot(y, 2)

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = Network.from_layer_sizes([2, 5, 2])

    losses, val_losses = model.train(
        x_train, y_train, x_val, y_val, lr=0.05, epochs=10000, log_every=100
    )

    print(f"Train accuracy: {accuracy(model, x_train, y_train)}")
    print(f"Validation accuracy: {accuracy(model, x_val, y_val)}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(list(losses.keys()), list(losses.values()), label="Training Loss")
    plt.plot(
        list(val_losses.keys()), list(val_losses.values()), label="Validation Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Draw samples and decision boundary
    granularity = 1000
    x1 = jnp.linspace(-1.5, 2.5, granularity)
    x2 = jnp.linspace(-1.5, 2.5, granularity)
    xx1, xx2 = jnp.meshgrid(x1, x2)
    x_grid = jnp.c_[xx1.ravel(), xx2.ravel()]
    y_grid = model(x_grid)[:, 0]
    y_grid = y_grid.reshape(xx1.shape)
    plt.figure(figsize=(10, 5))
    plt.contourf(xx1, xx2, y_grid, levels=50, cmap="RdBu", alpha=0.5)
    plt.scatter(x_train[:, 0], x_train[:, 1], c="blue", label="Train")
    plt.scatter(x_val[:, 0], x_val[:, 1], c="orange", label="Validation")
    plt.scatter(
        X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="red", label="Class 1", alpha=0.5
    )
    plt.scatter(
        X[y[:, 1] == 1, 0], X[y[:, 1] == 1, 1], c="green", label="Class 2", alpha=0.5
    )
    plt.title("Decision Boundary and Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()
