from __future__ import annotations

import math

import numpy as np

from gameboard import GomokuLogic


_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def action_to_coord(action: int, board_size: int) -> tuple[int, int]:
    return divmod(int(action), board_size)


def coord_to_action(row: int, col: int, board_size: int) -> int:
    return row * board_size + col


def candidate_actions(board: np.ndarray, radius: int = 2) -> np.ndarray:
    size = board.shape[0]
    occupied = np.argwhere(board != 0)
    if occupied.size == 0:
        center = size // 2
        return np.array([coord_to_action(center, center, size)], dtype=np.int64)

    candidate_mask = np.zeros((size, size), dtype=bool)
    for row, col in occupied:
        min_row = max(0, row - radius)
        max_row = min(size, row + radius + 1)
        min_col = max(0, col - radius)
        max_col = min(size, col + radius + 1)
        candidate_mask[min_row:max_row, min_col:max_col] = True

    empty_mask = board == 0
    coords = np.argwhere(candidate_mask & empty_mask)
    if coords.size == 0:
        coords = np.argwhere(empty_mask)
    actions = coords[:, 0] * size + coords[:, 1]
    return actions.astype(np.int64, copy=False)


def _would_win(board: np.ndarray, row: int, col: int, player: int) -> bool:
    logic = GomokuLogic(size=board.shape[0])
    logic.board = board.copy()
    success, msg = logic.step(row, col, player)
    return bool(success and msg == "Win")


def immediate_winning_actions(board: np.ndarray, player: int) -> list[int]:
    size = board.shape[0]
    winners: list[int] = []
    for action in candidate_actions(board, radius=1):
        row, col = action_to_coord(int(action), size)
        if _would_win(board, row, col, player):
            winners.append(int(action))
    return winners


def _line_score(board: np.ndarray, row: int, col: int, player: int) -> float:
    size = board.shape[0]
    total_score = 0.0
    for dr, dc in _DIRECTIONS:
        count = 1
        open_ends = 0

        for sign in (1, -1):
            step = 1
            while True:
                nr = row + sign * dr * step
                nc = col + sign * dc * step
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                    count += 1
                    step += 1
                    continue
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                    open_ends += 1
                break

        if count >= 5:
            total_score += 100000.0
        elif count == 4 and open_ends == 2:
            total_score += 18000.0
        elif count == 4 and open_ends == 1:
            total_score += 7000.0
        elif count == 3 and open_ends == 2:
            total_score += 2200.0
        elif count == 3 and open_ends == 1:
            total_score += 350.0
        elif count == 2 and open_ends == 2:
            total_score += 90.0
        else:
            total_score += count * (3.0 + open_ends)
    return total_score


def score_action(board: np.ndarray, action: int, player: int) -> float:
    size = board.shape[0]
    row, col = action_to_coord(action, size)
    if board[row, col] != 0:
        return -math.inf

    if _would_win(board, row, col, player):
        return 1_000_000.0
    if _would_win(board, row, col, -player):
        return 900_000.0

    center = (size - 1) / 2.0
    center_bias = size - (abs(row - center) + abs(col - center))

    board[row, col] = player
    attack = _line_score(board, row, col, player)
    board[row, col] = -player
    defense = _line_score(board, row, col, -player)
    board[row, col] = 0

    neighborhood = board[max(0, row - 1):min(size, row + 2), max(0, col - 1):min(size, col + 2)]
    local_density = float(np.count_nonzero(neighborhood))

    return attack + 0.92 * defense + 12.0 * center_bias + 18.0 * local_density


def heuristic_policy(board: np.ndarray, player: int) -> np.ndarray:
    size = board.shape[0]
    policy = np.zeros(size * size, dtype=np.float32)
    legal = candidate_actions(board)
    if legal.size == 0:
        return policy

    winning = immediate_winning_actions(board, player)
    if winning:
        policy[winning] = 1.0 / len(winning)
        return policy

    blocks = immediate_winning_actions(board, -player)
    if blocks:
        policy[blocks] = 1.0 / len(blocks)
        return policy

    scores = np.array([score_action(board, int(action), player) for action in legal], dtype=np.float64)
    scores -= scores.max()
    weights = np.exp(scores / 24.0)
    weights_sum = float(weights.sum())
    if weights_sum <= 0.0:
        policy[legal] = 1.0 / len(legal)
        return policy
    policy[legal] = weights / weights_sum
    return policy


def choose_heuristic_action(board: np.ndarray, player: int) -> int:
    policy = heuristic_policy(board, player)
    legal = np.flatnonzero(board.reshape(-1) == 0)
    if legal.size == 0:
        return -1
    best_index = legal[np.argmax(policy[legal])]
    return int(best_index)