import torch
import torch.nn as nn


class LowRankBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super(LowRankBlock, self).__init__()
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)

    def forward(self, x):
        return self.B(self.A(x))


class LowRankTransformBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super(LowRankTransformBlock, self).__init__()
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)

        self.T = nn.Linear(rank, rank)

    def forward(self, x):
        x = self.A(x)
        x = self.T(x)
        return self.B(x)


class LowRankDataLeftInitBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rank, low_rank_init, trainable=False):
        super(LowRankDataLeftInitBlock, self).__init__()
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)
        self.T = nn.Linear(rank, rank)

        with torch.no_grad():
            self.A.weight.copy_(low_rank_init)
            if self.A.bias is not None:
                self.A.bias.zero_()
        if not trainable:
            for param in self.A.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.A(x)
        x = self.T(x)
        return self.B(x)


class LowRankDataRightInitBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rank, low_rank_init, trainable=False):
        super(LowRankDataRightInitBlock, self).__init__()
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)
        self.T = nn.Linear(rank, rank)

        with torch.no_grad():
            self.B.weight.copy_(low_rank_init)
            if self.B.bias is not None:
                self.B.bias.zero_()
        if not trainable:
            for param in self.B.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.A(x)
        x = self.T(x)
        return self.B(x)


class LowRankDataInitBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        rank,
        low_rank_init_left,
        low_rank_init_right,
        trainable=False,
    ):
        super(LowRankDataInitBlock, self).__init__()
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)
        self.T = nn.Linear(rank, rank)

        with torch.no_grad():
            self.A.weight.copy_(low_rank_init_left)
            if self.A.bias is not None:
                self.A.bias.zero_()
            self.B.weight.copy_(low_rank_init_right)
            if self.B.bias is not None:
                self.B.bias.zero_()
        if not trainable:
            for param in self.A.parameters():
                param.requires_grad = False
            for param in self.B.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.A(x)
        x = self.T(x)
        return self.B(x)


class LowRankDataTransformInitBlock(nn.Module):
    def __init__(self, in_dim, out_dim, rank, low_rank_init, trainable=False):
        super(LowRankDataTransformInitBlock, self).__init__()
        self.A = nn.Linear(in_dim, rank)
        self.B = nn.Linear(rank, out_dim)

        with torch.no_grad():
            self.T = nn.Linear(rank, rank)
            self.T.weight.copy_(low_rank_init)
            if self.T.bias is not None:
                self.T.bias.zero_()
        if not trainable:
            for param in self.T.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.A(x)
        x = self.T(x)
        return self.B(x)
