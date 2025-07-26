import lightning as L
import torch
import torch.nn as nn


class Network(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int] = [64, 64],
        blocks: list = [nn.Linear, nn.Linear],
        metrics: dict = None,
    ):
        super(Network, self).__init__()
        layers = []
        prev_dim = in_dim
        for dim, block in zip(hidden_dims, blocks):
            layers.append(block(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.network = nn.Sequential(*layers)

        self.configure_metrics(metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)

        # Log additional metrics if provided
        y = y.argmax(dim=1)
        preds = logits.argmax(dim=1)
        for metric_name, metric_fn in self.metrics["train"].items():
            metric_fn.update(preds, y)
            self.log(f"train_{metric_name}", metric_fn, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("val_loss", loss)

        # Log additional metrics if provided
        y = y.argmax(dim=1)
        preds = logits.argmax(dim=1)
        for metric_name, metric_fn in self.metrics["val"].items():
            metric_fn.update(preds, y)
            self.log(f"val_{metric_name}", metric_fn, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("test_loss", loss)

        # Log additional metrics if provided
        y = y.argmax(dim=1)
        preds = logits.argmax(dim=1)
        for metric_name, metric_fn in self.metrics["test"].items():
            metric_fn.update(preds, y)
            self.log(f"test_{metric_name}", metric_fn, on_step=False, on_epoch=True)
        return loss

    def configure_metrics(self, metrics: dict):
        """Configure metrics for training, validation, and test."""
        self.metrics = (
            metrics if metrics is not None else {"train": {}, "val": {}, "test": {}}
        )
        # Make sure to register the metrics
        for phase in self.metrics:
            for metric_name, metric_fn in self.metrics[phase].items():
                self.add_module(f"{phase}_{metric_name}", metric_fn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_train_epoch_end(self):
        # Reset metrics at the end of each training epoch
        for _, metric_fn in self.metrics["train"].items():
            metric_fn.reset()

    def on_validation_epoch_end(self):
        # Reset metrics at the end of each validation epoch
        for _, metric_fn in self.metrics["val"].items():
            metric_fn.reset()

    def on_test_epoch_end(self):
        # Reset metrics at the end of each test epoch
        for _, metric_fn in self.metrics["test"].items():
            metric_fn.reset()
