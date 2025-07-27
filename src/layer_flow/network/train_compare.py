from dataclasses import dataclass

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

from layer_flow.clustering import kmpp
from layer_flow.data import DatasetFactory
from layer_flow.network.low_rank_network import (
    LowRankBlock,
    LowRankTransformBlock,
    LowRankDataInitBlock,
    LowRankDataLeftInitBlock,
    LowRankDataRightInitBlock,
    LowRankDataTransformInitBlock,
)
from layer_flow.network.network import Network
from layer_flow import CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR


def get_metrics(n_classes, task):
    return {
        "train": {
            "accuracy": Accuracy(task=task, num_classes=n_classes),
            "precision": Precision(task=task, num_classes=n_classes, average="macro"),
            "recall": Recall(task=task, num_classes=n_classes, average="macro"),
            "f1_score": F1Score(task=task, num_classes=n_classes, average="macro"),
        },
        "val": {
            "accuracy": Accuracy(task=task, num_classes=n_classes),
            "precision": Precision(task=task, num_classes=n_classes, average="macro"),
            "recall": Recall(task=task, num_classes=n_classes, average="macro"),
            "f1_score": F1Score(task=task, num_classes=n_classes, average="macro"),
        },
        "test": {
            "accuracy": Accuracy(task=task, num_classes=n_classes),
            "precision": Precision(task=task, num_classes=n_classes, average="macro"),
            "recall": Recall(task=task, num_classes=n_classes, average="macro"),
            "f1_score": F1Score(task=task, num_classes=n_classes, average="macro"),
        },
    }


@dataclass
class ModelConfig:
    name: str
    config: dict


def run_cross_validation(model_cfg: ModelConfig, dataset: Dataset, n_splits=5):
    torch.set_float32_matmul_precision("medium")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X = dataset.tensors[0]
    y = dataset.tensors[1]
    n_features = X.shape[1]
    n_classes = y.shape[1] if len(y.shape) > 1 else len(np.unique(y))

    all_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(y)), np.argmax(y, axis=1))
    ):
        print(f"Fold {fold + 1}/{n_splits}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=2048, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_subset, batch_size=2048, shuffle=False, num_workers=4
        )

        metrics = get_metrics(n_classes, "binary" if n_classes == 2 else "multiclass")
        fold_model = Network(
            in_dim=n_features, out_dim=n_classes, metrics=metrics, **model_cfg.config
        )

        trainer = L.Trainer(
            max_epochs=30,
            accelerator="auto",
            devices=1,
            logger=TensorBoardLogger(LOGS_DIR, name=f"{model_cfg.name}_fold{fold}"),
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=CHECKPOINTS_DIR,
                    filename="best-checkpoint",
                    save_top_k=1,
                    mode="min",
                ),
                EarlyStopping(
                    monitor="val_loss", patience=5, mode="min", verbose=False
                ),
            ],
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            log_every_n_steps=10,
        )

        trainer.fit(fold_model, train_loader, val_loader)

        val_result = trainer.validate(fold_model, val_loader, verbose=False)
        all_scores.append(val_result[0])

    return all_scores


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run cross-validation for different models.")
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Name of the dataset to use."
    )

    args = parser.parse_args()

    # Load MNIST dataset
    ds = DatasetFactory.create(args.dataset)
    full_train = ds.get_torch(split=False)

    def get_landmarks(k, d):
        """Generate k landmarks in d-dimensional space."""
        l_ds = ds
        if d < ds.n_features:
            l_ds = ds.subspace_embedding(n_components=d)
        raw_ds = l_ds.get_torch(split=False).tensors[0]
        # Pottentially leaking data here
        return kmpp(raw_ds, k, seed=42)

    # Run CV for all models
    hidden_dim = 1024
    rank = 32

    model = ModelConfig(
        name=f"{args.dataset}_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [nn.Linear],
        },
    )
    low_rank_model = ModelConfig(
        name=f"{args.dataset}_low_rank_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankBlock(in_dim, out_dim, rank=rank)
            ],
        },
    )
    low_rank_transform_model = ModelConfig(
        name=f"{args.dataset}_low_rank_transform_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankTransformBlock(
                    in_dim, out_dim, rank=rank
                )
            ],
        },
    )
    low_rank_data_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_init_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init_left=get_landmarks(in_dim, rank).T,
                    low_rank_init_right=get_landmarks(out_dim, rank),
                    trainable=True,
                )
            ],
        },
    )
    low_rank_data_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init_left=get_landmarks(in_dim, rank).T,
                    low_rank_init_right=get_landmarks(out_dim, rank),
                    trainable=False,
                )
            ],
        },
    )
    low_rank_data_left_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_left_init_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataLeftInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(in_dim, rank).T,
                    trainable=True,
                )
            ],
        },
    )
    low_rank_data_left_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_left_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataLeftInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(in_dim, rank).T,
                    trainable=False,
                )
            ],
        },
    )
    low_rank_data_right_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_right_init_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataRightInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(out_dim, rank),
                    trainable=True,
                )
            ],
        },
    )
    low_rank_data_right_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_right_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataRightInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(out_dim, rank),
                    trainable=False,
                )
            ],
        },
    )
    low_rank_data_transform_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_transform_init_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataTransformInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(rank, rank),
                    trainable=True,
                )
            ],
        },
    )
    low_rank_data_transform_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_transform_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim],
            "blocks": [
                lambda in_dim, out_dim: LowRankDataTransformInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(rank, rank),
                    trainable=False,
                )
            ],
        },
    )

    deep_model = ModelConfig(
        name=f"{args.dataset}_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [nn.Linear] * 3,
        },
    )
    deep_low_rank_model = ModelConfig(
        name=f"{args.dataset}_low_rank_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [lambda in_dim, out_dim: LowRankBlock(in_dim, out_dim, rank=rank)]
            * 3,
        },
    )
    deep_low_rank_transform_model = ModelConfig(
        name=f"{args.dataset}_low_rank_transform_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankTransformBlock(
                    in_dim, out_dim, rank=rank
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_init_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init_left=get_landmarks(in_dim, rank).T,
                    low_rank_init_right=get_landmarks(out_dim, rank),
                    trainable=True,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init_left=get_landmarks(in_dim, rank).T,
                    low_rank_init_right=get_landmarks(out_dim, rank),
                    trainable=False,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_left_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_left_init_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataLeftInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(in_dim, rank).T,
                    trainable=True,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_left_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_left_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataLeftInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(in_dim, rank).T,
                    trainable=False,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_right_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_right_init_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataRightInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(out_dim, rank),
                    trainable=True,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_right_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_right_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataRightInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(out_dim, rank),
                    trainable=False,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_transform_init_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_transform_init_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataTransformInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(rank, rank),
                    trainable=True,
                )
            ]
            * 3,
        },
    )
    deep_low_rank_data_transform_init_nontrainable_model = ModelConfig(
        name=f"{args.dataset}_low_rank_data_transform_init_nontrainable_network",
        config={
            "hidden_dims": [hidden_dim] * 3,
            "blocks": [
                lambda in_dim, out_dim: LowRankDataTransformInitBlock(
                    in_dim,
                    out_dim,
                    rank=rank,
                    low_rank_init=get_landmarks(rank, rank),
                    trainable=False,
                )
            ]
            * 3,
        },
    )

    all_models = [
        model,
        low_rank_model,
        low_rank_transform_model,
        low_rank_data_init_model,
        low_rank_data_init_nontrainable_model,
        low_rank_data_left_init_model,
        low_rank_data_left_init_nontrainable_model,
        low_rank_data_right_init_model,
        low_rank_data_right_init_nontrainable_model,
        low_rank_data_transform_init_model,
        low_rank_data_transform_init_nontrainable_model,
        deep_model,
        deep_low_rank_model,
        deep_low_rank_transform_model,
        deep_low_rank_data_init_model,
        deep_low_rank_data_init_nontrainable_model,
        deep_low_rank_data_left_init_model,
        deep_low_rank_data_left_init_nontrainable_model,
        deep_low_rank_data_right_init_model,
        deep_low_rank_data_right_init_nontrainable_model,
        deep_low_rank_data_transform_init_model,
        deep_low_rank_data_transform_init_nontrainable_model,
    ]

    results = []
    for model_cfg in all_models:
        print(f"Running cross-validation for {model_cfg.name}...")
        print("-" * 50)
        results.append(run_cross_validation(model_cfg, full_train, n_splits=5))

    # Compare average scores
    def summarize(results):
        metrics = results[0].keys()
        return {m: np.mean([r[m] for r in results]) for m in metrics}

    summary = {
        model_cfg.name: summarize(r) for model_cfg, r in zip(all_models, results)
    }

    print("\nCross-validation results:")
    for model_name, scores in summary.items():
        print(f"{model_name}:")
        for metric, score in scores.items():
            print(f"\t{metric}: {score:.4f}")
        print("-" * 50)

    # Write results to CSV
    df = pd.DataFrame(summary).T
    df.index.name = "model"
    df.to_csv(RESULTS_DIR / f"{args.dataset}_cross_validation_results.csv")
    print(
        f"Results saved to {RESULTS_DIR / f'{args.dataset}_cross_validation_results.csv'}"
    )

    # Save plot
    df = pd.read_csv(RESULTS_DIR / f"{args.dataset}_cross_validation_results.csv")
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(df["model"], df["val_loss"], marker="o")
    plt.title("Validation Loss")
    plt.xticks(rotation=45, ha="right")
    plt.subplot(2, 2, 2)
    plt.plot(df["model"], df["val_accuracy"], marker="o")
    plt.title("Validation Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.subplot(2, 2, 3)
    plt.plot(df["model"], df["val_precision"], marker="o")
    plt.title("Validation Precision")
    plt.xticks(rotation=45, ha="right")
    plt.subplot(2, 2, 4)
    plt.plot(df["model"], df["val_recall"], marker="o")
    plt.title("Validation Recall")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{args.dataset}_cross_validation_results.png")
