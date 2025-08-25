import torch
import torch.nn as nn


class IntermediateExtractor(nn.Module):
    """
    A neural network module that returns both the final output and all intermediate representations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        batch_norm: bool = False,
    ):
        super(IntermediateExtractor, self).__init__()
        layers = []
        self.names = [] # For naming layers
        prev_size = input_size
        idx = 0
        for hidden in hidden_sizes:
            linear = nn.Linear(prev_size, hidden)
            if batch_norm:
                linear = nn.Sequential(linear, nn.BatchNorm1d(hidden))
            layers.append(linear)
            self.names.append(f"Layer {idx}: Linear {prev_size}->{hidden}")
            layers.append(nn.ReLU())
            self.names.append(f"Layer {idx}: ReLU")
            prev_size = hidden
            idx += 1

        # Final output layer (no batch norm or activation)
        layers.append(nn.Linear(prev_size, output_size))
        self.names.append(f"Layer {idx}: Linear {prev_size}->{output_size}")
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        intermediates = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            key = self.names[idx]
            intermediates[key] = x
        return x, intermediates
