from __future__ import annotations

import numpy as np

from Mehuls_agent.state import GomokuState


def encode_state(state: GomokuState) -> np.ndarray:
    board = state.board
    current = (board == state.current_player).astype(np.float32)
    opponent = (board == -state.current_player).astype(np.float32)
    color_plane = np.full(board.shape, 1.0 if state.current_player == 1 else 0.0, dtype=np.float32)
    last_move_plane = np.zeros(board.shape, dtype=np.float32)
    if state.last_move is not None:
        row, col = state.last_move
        last_move_plane[row, col] = 1.0
    return np.stack([current, opponent, color_plane, last_move_plane], axis=0)


def _rot90_features(features: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(features, k=k, axes=(-2, -1)).copy()


def _rot90_policy(policy: np.ndarray, board_size: int, k: int) -> np.ndarray:
    board = policy.reshape(board_size, board_size)
    return np.rot90(board, k=k).reshape(-1).copy()


def augment_example(features: np.ndarray, policy: np.ndarray, board_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    augmented = []
    for rotation in range(4):
        rotated_features = _rot90_features(features, rotation)
        rotated_policy = _rot90_policy(policy, board_size, rotation)
        augmented.append((rotated_features, rotated_policy))

        flipped_features = np.flip(rotated_features, axis=-1).copy()
        flipped_policy = np.flip(rotated_policy.reshape(board_size, board_size), axis=-1).reshape(-1).copy()
        augmented.append((flipped_features, flipped_policy))
    return augmented