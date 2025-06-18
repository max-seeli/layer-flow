from jax import numpy as jnp
from typing import Dict, Tuple
from tqdm import trange
from layer_flow.network import Network, LowRankNetwork

def train(
        model: Network,
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
            model (Network): The network to be trained.
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
            model.update(x_train, y_train, lr)
            if epoch % log_every == 0 or epoch in [1, epochs]:
                loss = model.loss(x_train, y_train)
                val_loss = model.loss(x_val, y_val)
                if verbose:
                    print(f"@[{epoch}]: train loss {loss}; val loss {val_loss}")
                else:
                    pbar.set_postfix({"epoch": epoch, "train loss": loss, "val loss": val_loss})
                    pbar
                losses[epoch] = loss
                val_losses[epoch] = val_loss
        return losses, val_losses

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
    from layer_flow.data import DatasetFactory
    import matplotlib.pyplot as plt

    ds = DatasetFactory.create("cifar10")
    X, y = ds.X, ds.y

    x_train, x_val, y_train, y_val = ds.X_train, ds.X_val, ds.y_train, ds.y_val

    in_dim = ds.n_features
    out_dim = ds.n_classes

    #model = Network.from_layer_sizes([in_dim, in_dim, out_dim], seed=0)
    #model = LowRankNetwork.from_depth_rank(2, 50, in_dim=in_dim, out_dim=out_dim, seed=0)
    model = LowRankNetwork.from_coreset_depth_rank(x_train, 2, 50, in_dim=in_dim, out_dim=out_dim, seed=0)


    losses, val_losses = train(
        model, x_train, y_train, x_val, y_val, lr=0.3, epochs=150, log_every=5
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

