from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        x = self.activation(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int,
        input_planes: int = 4,
        channels: int = 96,
        residual_blocks: int = 6,
        value_hidden_dim: int = 128,
    ):
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(residual_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_linear = nn.Linear(2 * board_size * board_size, board_size * board_size)

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_linear1 = nn.Linear(board_size * board_size, value_hidden_dim)
        self.value_linear2 = nn.Linear(value_hidden_dim, 1)
        self.value_activation = nn.ReLU(inplace=True)
        self.value_tanh = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(inputs))

        policy = self.policy_head(x)
        policy = policy.view(policy.shape[0], -1)
        policy_logits = self.policy_linear(policy)

        value = self.value_head(x)
        value = value.view(value.shape[0], -1)
        value = self.value_activation(self.value_linear1(value))
        value = self.value_tanh(self.value_linear2(value))
        return policy_logits, value.squeeze(-1)