import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from layer_flow import RESULTS_DIR
from layer_flow.clustering import (
    compute_clustering_cost,
    compute_mean_uniformity,
    coreset_tree,
    kmpp,
    lloyd,
)
from layer_flow.data import DatasetFactory
from layer_flow.network_cluster_analysis.inter_rep_net import IntermediateExtractor


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def make_cluster_fn(y_labels: np.ndarray, k: int):
    """Returns closures to compute clustering cost and mean uniformity on numpy arrays."""

    def cluster_fn(X_np: np.ndarray) -> float:
        if X_np.ndim > 2:  # Flatten if necessary
            X_np = X_np.reshape(X_np.shape[0], -1)

        centers = lloyd(X_np, kmpp(X_np, k=k), max_iter=30, tol=1e-2, verbose=True)

        cost = compute_clustering_cost(X_np, centers)
        uniformity = compute_mean_uniformity(X_np, y_labels, centers)
        return cost, uniformity

    return cluster_fn

def make_explained_variance_fn(X_np: np.ndarray):
    """Returns a closure to compute explained variance on numpy arrays."""
    if X_np.ndim > 2:  # Flatten if necessary
        X_np = X_np.reshape(X_np.shape[0], -1)
    total_variance = np.var(X_np, axis=0).sum()

    def explained_variance_fn(X_np_layer: np.ndarray) -> float:
        if X_np_layer.ndim > 2:  # Flatten if necessary
            X_np_layer = X_np_layer.reshape(X_np_layer.shape[0], -1)
        layer_variance = np.var(X_np_layer, axis=0).sum()
        return layer_variance / total_variance

    return explained_variance_fn


def plot_metrics(
    costs: np.ndarray,
    uniformities: np.ndarray,
    explained_variances: np.ndarray,
    accuracies: np.ndarray,
    initial_cost: float,
    initial_uniformity: float,
    out_path: Path,
    layer_names: List[str] = None,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Clustering Cost
    for i in range(costs.shape[1]):
        label = f"Layer {i}" if layer_names is None else layer_names[i]
        axs[0].plot(costs[:, i], label=label)
    #axs[0].set_yscale("log")
    axs[0].axhline(y=initial_cost, linestyle="--", label="Initial Cost")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Clustering Cost")
    axs[0].set_title("Clustering Cost per Layer")
    axs[0].legend()

    # Mean Uniformity
    for i in range(uniformities.shape[1]):
        label = f"Layer {i}" if layer_names is None else layer_names[i]
        axs[1].plot(uniformities[:, i], label=label)
    axs[1].axhline(y=initial_uniformity, linestyle="--", label="Initial Uniformity")

    ax2 = axs[1].twinx()
    ax2.plot(accuracies, color="gray", linestyle=":", label="Accuracy")

    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Mean Uniformity")
    ax2.set_ylabel("Accuracy")
    axs[1].set_title("Mean Uniformity per Layer & Accuracy")

    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Explained Variance
    for i in range(explained_variances.shape[1]):
        label = f"Layer {i}" if layer_names is None else layer_names[i]
        axs[2].plot(explained_variances[:, i], label=label)
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Explained Variance")
    axs[2].set_title("Explained Variance per Layer")
    axs[2].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def evaluate_layers(
    intermediates: Dict[str, torch.Tensor], intermediates_order: List[str], cluster_fn, explained_variance_fn
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cost and uniformity for each intermediate layer (order preserved)."""
    layer_keys = intermediates_order
    layer_costs = np.zeros(len(layer_keys), dtype=float)
    layer_uniformities = np.zeros(len(layer_keys), dtype=float)
    layer_explained_variances = np.zeros(len(layer_keys), dtype=float)

    for i, key in enumerate(layer_keys):
        X_np = to_numpy(intermediates[key])
        layer_costs[i], layer_uniformities[i] = cluster_fn(X_np)
        layer_explained_variances[i] = explained_variance_fn(X_np)

    return layer_costs, layer_uniformities, layer_explained_variances


def train_and_track(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    k_classes: int,
    epochs: int = 30,
    lr: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Trains the model while tracking clustering cost and mean uniformity of intermediate reps.
    Returns (costs_over_time, uniformities_over_time, initial_cost, initial_uniformity).
    """

    # Prepare metrics
    y_labels = to_numpy(y_tensor)
    if y_labels.ndim > 1:
        y_labels = np.argmax(y_labels, axis=1)
    y_labels_tensor = torch.tensor(y_labels, dtype=torch.long)

    cluster_fn = make_cluster_fn(y_labels=y_labels, k=k_classes)
    data_cost, data_uniformity = cluster_fn(to_numpy(X_tensor))

    explained_variance_fn = make_explained_variance_fn(to_numpy(X_tensor))

    # Forward once to know number of intermediate layers
    with torch.inference_mode():
        _, intermediates = model(X_tensor)
        n_layers = len(intermediates)

    costs = np.zeros((epochs + 1, n_layers), dtype=float)
    uniformities = np.zeros((epochs + 1, n_layers), dtype=float)
    explained_variances = np.zeros((epochs + 1, n_layers), dtype=float)
    accuracies = np.zeros(epochs + 1, dtype=float)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        y_hat, intermediates = model(X_tensor)
        loss = criterion(y_hat, y_tensor)
        loss.backward()
        optimizer.step()

        # This adds the metrics for the previous epoch, as they are computed before the update
        c_e, u_e, ev_e = evaluate_layers(intermediates, model.names, cluster_fn, explained_variance_fn)
        costs[epoch] = c_e
        uniformities[epoch] = u_e
        explained_variances[epoch] = ev_e
        accuracies[epoch + 1] = (
            (y_hat.cpu().argmax(dim=1) == y_labels_tensor).float().mean().item()
        )
        print(f"Accuracy at epoch {epoch}: {accuracies[epoch]:.4f}")

    # Final metrics after training
    with torch.inference_mode():
        _, intermediates = model(X_tensor)
        c_final, u_final, ev_final = evaluate_layers(intermediates, model.names, cluster_fn, explained_variance_fn)
        costs[epochs] = c_final
        uniformities[epochs] = u_final
        explained_variances[epochs] = ev_final

    return costs, uniformities, explained_variances, accuracies, data_cost, data_uniformity


def main(epochs: int = 30, lr: float = 1e-2, hidden_sizes: List[int] = [32, 32]):
    device = get_device()
    print("Using device:", device)

    # Data
    ds = DatasetFactory.create("mnist")
    X, y = ds.X, ds.y

    print("Dataset created. Statistics:")
    print(" #samples:", ds.n_samples)
    print(" #classes:", ds.n_classes)
    print(" #features:", ds.n_features)
    print("=" * 40)

    # Model
    model = IntermediateExtractor(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=ds.n_classes,
        batch_norm=True,
    ).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Initial forward to show shapes
    with torch.inference_mode():
        y_hat, intermediates = model(X_tensor)
    print("Logits shape:", tuple(y_hat.shape))
    print("Intermediate layers:", list(intermediates.keys()))
    for k, v in intermediates.items():
        print(f" {k}: {tuple(v.shape)}")

    # Train & track metrics
    costs, uniformities, explained_variances, accuracies, data_cost, data_uniformity = train_and_track(
        model=model,
        X_tensor=X_tensor,
        y_tensor=y_tensor,
        k_classes=ds.n_classes,
        epochs=epochs,
        lr=lr,
    )

    # Plot
    out_path = Path(RESULTS_DIR) / "cluster_analysis_results.png"
    plot_metrics(costs, uniformities, explained_variances, accuracies, data_cost, data_uniformity, out_path, layer_names=model.names)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intermediate Representation Cluster Analysis"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[32, 32])
    args = parser.parse_args()

    main(epochs=args.epochs, lr=args.lr, hidden_sizes=args.hidden_sizes)
