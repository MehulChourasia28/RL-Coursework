"""
Gomoku heuristics: threat detection, move scoring, and heuristic opponent.
"""

import random
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level line analysis
# ---------------------------------------------------------------------------

def _count_line(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int
                ) -> Tuple[int, bool, bool]:
    """
    Count consecutive *player* stones through (row, col) in direction (dr, dc).
    Returns (count, open_back, open_fwd) where open_* means the line end is empty.
    The stone at (row, col) must already be player's stone (or we test it).
    """
    size = board.shape[0]
    count = 1  # the stone itself

    # Forward
    fwd_open = False
    for i in range(1, 6):
        r, c = row + dr * i, col + dc * i
        if 0 <= r < size and 0 <= c < size:
            if board[r][c] == player:
                count += 1
            elif board[r][c] == 0:
                fwd_open = True
                break
            else:
                break
        else:
            break

    # Backward
    bwd_open = False
    for i in range(1, 6):
        r, c = row - dr * i, col - dc * i
        if 0 <= r < size and 0 <= c < size:
            if board[r][c] == player:
                count += 1
            elif board[r][c] == 0:
                bwd_open = True
                break
            else:
                break
        else:
            break

    return count, bwd_open, fwd_open


def _line_score(count: int, open_back: bool, open_fwd: bool) -> float:
    """Assign a score to a line pattern."""
    opens = int(open_back) + int(open_fwd)
    if count >= 5:
        return 100_000.0
    if count == 4:
        return 10_000.0 if opens >= 1 else 100.0
    if count == 3:
        return 1_000.0 if opens == 2 else 50.0
    if count == 2:
        return 50.0 if opens == 2 else 5.0
    return 1.0 if opens >= 1 else 0.0


def _score_move(board: np.ndarray, row: int, col: int, player: int) -> float:
    """Score placing player at (row, col) — stone must NOT be on board yet."""
    board[row][col] = player
    score = 0.0
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count, ob, of = _count_line(board, row, col, dr, dc, player)
        score += _line_score(count, ob, of)
    board[row][col] = 0
    return score


# ---------------------------------------------------------------------------
# Win / block detection
# ---------------------------------------------------------------------------

def _check_win_at(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """Return True if (row, col) is part of a 5-in-a-row for player."""
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count, _, _ = _count_line(board, row, col, dr, dc, player)
        if count >= 5:
            return True
    return False


def get_winning_move(board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
    """Return any immediate winning move for player, or None."""
    size = board.shape[0]
    for r in range(size):
        for c in range(size):
            if board[r][c] == 0:
                board[r][c] = player
                wins = _check_win_at(board, r, c, player)
                board[r][c] = 0
                if wins:
                    return (r, c)
    return None


# ---------------------------------------------------------------------------
# Candidate move generation
# ---------------------------------------------------------------------------

def get_candidate_moves(board: np.ndarray, radius: int = 2) -> List[Tuple[int, int]]:
    """Empty cells within *radius* of any occupied cell."""
    size = board.shape[0]
    occupied = np.argwhere(board != 0)

    if len(occupied) == 0:
        return [(size // 2, size // 2)]

    candidates: set = set()
    for r, c in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == 0:
                    candidates.add((nr, nc))
    return list(candidates)


# ---------------------------------------------------------------------------
# Heuristic opponent
# ---------------------------------------------------------------------------

def heuristic_move(board: np.ndarray, player: int) -> Tuple[int, int]:
    """
    Pick a move for *player* using a threat-based heuristic:
      1. Win immediately if possible.
      2. Block opponent's immediate win.
      3. Score candidate cells and pick the best.
    """
    opponent = -player

    # 1. Win now
    move = get_winning_move(board, player)
    if move is not None:
        return move

    # 2. Block opponent win
    move = get_winning_move(board, opponent)
    if move is not None:
        return move

    # 3. Score candidates
    candidates = get_candidate_moves(board, radius=2)
    if not candidates:
        empty = list(zip(*np.where(board == 0)))
        if not empty:
            raise RuntimeError("No empty cells left on board")
        return tuple(map(int, random.choice(empty)))

    best_score = -1.0
    best_moves = []
    for r, c in candidates:
        atk = _score_move(board, r, c, player)
        def_ = _score_move(board, r, c, opponent)
        score = atk * 1.1 + def_        # slightly prefer attack
        if score > best_score:
            best_score = score
            best_moves = [(r, c)]
        elif score == best_score:
            best_moves.append((r, c))

    return random.choice(best_moves)
