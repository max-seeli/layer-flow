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
        prev_size = input_size
        for hidden in hidden_sizes:
            linear = nn.Linear(prev_size, hidden)
            if batch_norm:
                linear = nn.Sequential(linear, nn.BatchNorm1d(hidden))
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_size = hidden

        # Final output layer (no batch norm or activation)
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        intermediates = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            key = f"layer_{idx}_{layer.__class__.__name__}"
            intermediates[key] = x
        return x, intermediates
