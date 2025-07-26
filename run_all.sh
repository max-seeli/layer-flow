#!/bin/bash

nohup uv run python -u src/layer_flow/network/train_compare.py \
    --dataset "mnist" > logs/mnist_train.log 2>&1 &
nohup uv run python -u src/layer_flow/network/train_compare.py \
    --dataset "fashion_mnist" > logs/fashion_mnist_train.log 2>&1 &
nohup uv run python -u src/layer_flow/network/train_compare.py \
    --dataset "cifar10" > logs/cifar10_train.log 2>&1 &
