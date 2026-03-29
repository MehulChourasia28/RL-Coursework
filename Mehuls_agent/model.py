"""
Policy-Value network for AlphaZero-style Gomoku.

Input:  (B, 3, size, size)
  ch0 = current player's stones  (1 where board == player)
  ch1 = opponent's stones        (1 where board == -player)
  ch2 = ones                     (bias plane)

Outputs:
  log_policy : (B, size*size)  — log-probabilities over moves
  value      : (B,)            — win probability in [-1, +1]
                                 +1 = current player wins
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State encoding (used by both MCTS and training)
# ---------------------------------------------------------------------------

def encode_state(board: np.ndarray, player: int) -> np.ndarray:
    """
    Return (3, size, size) float32 tensor from the current player's perspective.
      ch0  own stones   (1 where board == player)
      ch1  opp stones   (1 where board == -player)
      ch2  ones         (bias plane)
    """
    own  = (board == player).astype(np.float32)
    opp  = (board == -player).astype(np.float32)
    ones = np.ones_like(own)
    return np.stack([own, opp, ones], axis=0)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


# ---------------------------------------------------------------------------
# PolicyValueNet
# ---------------------------------------------------------------------------

class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int = 9,
        in_channels: int = 3,
        channels: int = 64,
        num_res_blocks: int = 4,
        value_fc_hidden: int = 64,
    ):
        super().__init__()
        self.board_size = board_size
        n = board_size * board_size

        # Shared backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head: 2-filter conv → FC → log_softmax
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * n, n)

        # Value head: 1-filter conv → FC(64) → FC(1) → tanh
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(n, value_fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_fc_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, size, size)
        Returns:
            log_policy : (B, size*size)
            value      : (B,)
        """
        h = self.trunk(self.stem(x))

        p = self.policy_conv(h).flatten(1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        v = self.value_conv(h).flatten(1)
        value = self.value_fc(v).squeeze(1)

        return log_policy, value
