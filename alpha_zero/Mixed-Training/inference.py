"""
Public predict_move API for the AlphaZero Gomoku agent.
Falls back to heuristic if no checkpoint exists.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch

from .heuristics import heuristic_move

BOARD_SIZE      = 9
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
N_SIMS_PLAY     = 400        # simulations per move during human play

_mcts   = None
_device = None


def load_agent():
    """Lazy-load the MCTS agent. Returns None if no checkpoint found."""
    global _mcts, _device
    if _mcts is not None:
        return _mcts
    if not os.path.exists(CHECKPOINT_PATH):
        return None

    from .model import PolicyValueNet
    from .mcts  import MCTS

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt    = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)

    # Only load AlphaZero checkpoints (have 'config' key)
    if "config" not in ckpt:
        return None

    cfg = ckpt["config"]
    net = PolicyValueNet(
        board_size     = cfg.get("board_size",     BOARD_SIZE),
        channels       = cfg.get("channels",       64),
        num_res_blocks = cfg.get("num_res_blocks", 4),
    ).to(_device)

    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["net"].items()}
    net.load_state_dict(sd)
    net.eval()

    _mcts = MCTS(net, _device, n_simulations=N_SIMS_PLAY, dirichlet_epsilon=0.0)
    ckpt_iter = ckpt.get("iteration", "?")
    print(f"[Mehuls_agent] Using AlphaZero network (iter={ckpt_iter}, sims={N_SIMS_PLAY}, device={_device})")
    return _mcts


def predict_move(board_state: np.ndarray, player: int) -> Tuple[int, int]:
    """
    Return the best move as (x, y) = (col, row).

    Args:
        board_state : (size, size) numpy array, +1=black, -1=white, 0=empty
        player      : +1 or -1, the player to move

    Returns:
        (col, row)  — x=col, y=row, matching gomoku_human_vs_ai.py convention
    """
    agent = load_agent()
    board_size = board_state.shape[0]
    if agent is None or board_size != _mcts.net.board_size:
        trained_on = _mcts.net.board_size if agent else "N/A"
        print(f"[Mehuls_agent] Using HEURISTIC (network trained on {trained_on}x{trained_on}, got {board_size}x{board_size})")
        row, col = heuristic_move(board_state, player)
        return int(col), int(row)

    agent.reset()   # always start fresh — no persistent state between calls
    policy = agent.get_policy(board_state, player, temperature=0.0, add_noise=False)

    # Mask illegal moves
    valid  = (board_state.flatten() == 0).astype(np.float32)
    policy = policy * valid

    action = int(np.argmax(policy))
    row    = action // board_state.shape[0]
    col    = action  % board_state.shape[0]
    return int(col), int(row)   # x=col, y=row
