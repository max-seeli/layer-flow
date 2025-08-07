import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from layer_flow.clustering import compute_clustering_cost, coreset_tree, compute_mean_uniformity
from layer_flow.data import DatasetFactory
from layer_flow.network_cluster_analysis.inter_rep_net import \
    IntermediateExtractor

ds = DatasetFactory.create("mnist") #, n=1000, dim=5, num_blobs=10)
X, y = ds.X, ds.y
y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

print("Dataset created. Statistics:")
print(" #samples:", ds.n_samples)
print(" #classes:", ds.n_classes)
print(" #features:", ds.n_features)
print("=" * 40)

print("Computing clustering cost...")
cost = lambda X, k=ds.n_classes: compute_clustering_cost(X, coreset_tree(X, k=k))
print("Clustering cost:", cost(X))

print("Computing mean uniformity...")
uniformity = lambda X, k=ds.n_classes: compute_mean_uniformity(X, y_labels, coreset_tree(X, k=k))
print("Mean uniformity:", uniformity(X))

model = IntermediateExtractor(input_size=X.shape[1], hidden_sizes=[32, 32], output_size=ds.n_classes)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
y_hat, intermediates = model(X_tensor)
print(y_hat.shape)
print(intermediates.keys())
print(intermediates["layer_0_Linear"].shape)

epochs = 30
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

initial_cost = cost(X_tensor.detach().numpy())
costs = np.zeros((epochs + 1, len(intermediates)))

initial_uniformity = uniformity(X_tensor.detach().numpy())
uniformities = np.zeros((epochs + 1, len(intermediates)))

for i, (key, value) in enumerate(intermediates.items()):
    print(f"{key}: {value.shape}")
    c = cost(value.detach().numpy())
    costs[0, i] = c

    u = uniformity(value.detach().numpy())
    uniformities[0, i] = u

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    y_hat, intermediates = model(X_tensor)
    loss = criterion(y_hat, y_tensor)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    loss.backward()
    optimizer.step()

    for i, (key, value) in enumerate(intermediates.items()):
        c = cost(value.detach().numpy())
        costs[epoch + 1, i] = c

        u = uniformity(value.detach().numpy())
        uniformities[epoch + 1, i] = u


# Plot the costs
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot Clustering Cost
for i in range(costs.shape[1]):
    axs[0].plot(costs[:, i], label=f"Layer {i}")
axs[0].axhline(y=initial_cost, color='r', linestyle='--', label='Initial Cost')
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Clustering Cost")
axs[0].set_title("Clustering Cost per Layer")
axs[0].legend()

# Plot Mean Uniformity
for i in range(uniformities.shape[1]):
    axs[1].plot(uniformities[:, i], label=f"Layer {i}")
axs[1].axhline(y=initial_uniformity, color='r', linestyle='--', label='Initial Uniformity')
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Mean Uniformity")
axs[1].set_title("Mean Uniformity per Layer")
axs[1].legend()

plt.tight_layout()
plt.show()
