from layer_flow.network.torch.network import Network

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

from layer_flow.data import DatasetFactory
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Example usage

    ds = DatasetFactory.create("mnist")
    task = "binary" if ds.n_classes == 2 else "multiclass"
    metrics = {
        "train": {
            "accuracy": Accuracy(task=task, num_classes=ds.n_classes),
            "precision": Precision(task=task, num_classes=ds.n_classes, average='macro'),
            "recall": Recall(task=task, num_classes=ds.n_classes, average='macro'),
            "f1_score": F1Score(task=task, num_classes=ds.n_classes, average='macro'),
        },
        "val": {
            "accuracy": Accuracy(task=task, num_classes=ds.n_classes),
            "precision": Precision(task=task, num_classes=ds.n_classes, average='macro'),
            "recall": Recall(task=task, num_classes=ds.n_classes, average='macro'),
            "f1_score": F1Score(task=task, num_classes=ds.n_classes, average='macro'),
        },
        "test": {
            "accuracy": Accuracy(task=task, num_classes=ds.n_classes),
            "precision": Precision(task=task, num_classes=ds.n_classes, average='macro'),
            "recall": Recall(task=task, num_classes=ds.n_classes, average='macro'),
            "f1_score": F1Score(task=task, num_classes=ds.n_classes, average='macro'),
        }
    }


    model = Network(in_dim=784, out_dim=10, hidden_dims=[128, 64], metrics=metrics)
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",  # Use GPU if available
        devices=1,  # Use 1 GPU
        logger=TensorBoardLogger("logs/"),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="checkpoints/",
                filename="best-checkpoint",
                save_top_k=1,
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
                verbose=True,
            ),
        ],
        enable_progress_bar=True,
        profiler="simple",  # Use a simple profiler for performance monitoring
    )
    
    # Assuming you have a DataLoader for training and validation
    ds = ds.get_torch(split=True)

    train_loader = DataLoader(ds["train"], batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(ds["val"], batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(ds["test"], batch_size=32, shuffle=False, num_workers=4)


    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    