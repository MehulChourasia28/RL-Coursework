"""
Standardised predict interface for cross-group testing.

Usage:
    from alpha_zero.predict import predict

    # board_state: 9x9 numpy array — accepts BOTH formats:
    #   0/1/2    with current_player 1 (black) or 2 (white)
    #   0/+1/-1  with current_player +1 (black) or -1 (white)
    row, col = predict(board_state, current_player)

IMPORTANT NOTES ON STRENGTH (tree reuse):
    Normally in a full game, MCTS keeps its search tree between moves.
    When the opponent plays move X, we jump to the subtree we already
    explored for X — so the next search starts with prior work "for free".
    This gives roughly 10-20% more effective search depth at the same
    simulation count.

    This function is STATELESS — each call builds a fresh tree from
    scratch, so that prior work is lost. This is an intentional tradeoff:
    a clean predict(board, player) -> (row, col) API that anyone can call
    without managing MCTS state, at the cost of slightly weaker play.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    from alpha_zero.az_gomuku5 import (
        AZNet, Gomoku, Node, mcts,
        BOARD, N_SIMS, DEVICE,
    )
except ModuleNotFoundError:
    from az_gomuku5 import (
        AZNet, Gomoku, Node, mcts,
        BOARD, N_SIMS, DEVICE,
    )

# ── Resolve default weights path relative to this file ───────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_DIR, "..", "models_az5", "az_iter0150.pt")

# ── Module-level cache so the model is loaded only once ──────────────────────
_model: AZNet | None = None
_loaded_path: str | None = None


def _load_model(weights_path: str | None = None) -> AZNet:
    """Load (or return cached) AZNet from a checkpoint or raw state-dict file."""
    global _model, _loaded_path

    path = os.path.normpath(weights_path or _DEFAULT_WEIGHTS)
    if _model is not None and _loaded_path == path:
        return _model

    net = AZNet().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # Support both full checkpoints (with 'model' key) and plain state-dicts
    if isinstance(ckpt, dict) and "model" in ckpt:
        net.load_state_dict(ckpt["model"])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    _model = net
    _loaded_path = path
    return net


def predict(board_state: np.ndarray,
            current_player: int = 1,
            weights_path: str | None = None,
            n_sims: int = N_SIMS) -> tuple[int, int]:
    """
    Standardised prediction function.

    Args:
        board_state:    9x9 numpy array (0=empty, 1=black, 2=white).
        current_player: Which player is to move (1=black, 2=white).
        weights_path:   Optional path to model weights (defaults to latest checkpoint).
        n_sims:         MCTS simulations (default 400). Lower = faster but weaker.

    Returns:
        (row, col) tuple — the chosen move.
    """
    net = _load_model(weights_path)

    # Normalise board representation: accept both 0/1/2 and 0/+1/-1
    board = np.array(board_state, dtype=np.float64).reshape(BOARD, BOARD)
    if np.any(board < 0):                       # 0/+1/-1 format
        normalised = np.zeros((BOARD, BOARD), dtype=np.int8)
        normalised[board > 0]  = 1              # +1 → black
        normalised[board < 0]  = 2              # -1 → white
        if current_player == -1:
            current_player = 2
        elif current_player == 1:
            current_player = 1                  # already correct
    else:                                       # 0/1/2 format
        normalised = board.astype(np.int8)

    # Reconstruct a Gomoku game object from the raw board
    game = Gomoku()
    game.board = normalised
    game.player = current_player
    game.n_moves = int(np.count_nonzero(game.board))

    # We need a valid last_moves entry so terminal() can check for wins.
    # Since we don't know the true move order, scan for any opponent stone
    # and use it as a placeholder (only matters for win-check on that square).
    opponent = 3 - current_player
    opp_positions = list(zip(*np.where(game.board == opponent)))
    if opp_positions:
        lr, lc = opp_positions[-1]
        game.last_moves = [(lr * BOARD + lc, opponent)]

    # Run MCTS from this position
    root = Node(prior=1.0)
    pi = mcts(game, net, root, root_noise=False, n_sims=n_sims)

    action = int(np.argmax(pi))
    row, col = divmod(action, BOARD)
    return (row, col)


if __name__ == "__main__":
    # Quick smoke test with an empty board
    board = np.zeros((9, 9), dtype=int)
    move = predict(board, current_player=1)
    print(f"Predicted move on empty board: row={move[0]}, col={move[1]}")
