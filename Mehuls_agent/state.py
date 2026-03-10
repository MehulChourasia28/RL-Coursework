from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gameboard import GomokuLogic


@dataclass(slots=True)
class GomokuState:
    size: int
    board: np.ndarray
    current_player: int = 1
    last_move: tuple[int, int] | None = None
    winner: int = 0
    terminal: bool = False

    @classmethod
    def new_game(cls, size: int) -> "GomokuState":
        return cls(size=size, board=np.zeros((size, size), dtype=np.int8))

    def clone(self) -> "GomokuState":
        return GomokuState(
            size=self.size,
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            terminal=self.terminal,
        )

    def legal_actions(self) -> np.ndarray:
        empty = np.flatnonzero(self.board.reshape(-1) == 0)
        return empty.astype(np.int64, copy=False)

    def apply_action(self, action: int) -> "GomokuState":
        row, col = divmod(int(action), self.size)
        logic = GomokuLogic(size=self.size)
        logic.board = self.board.copy()
        logic.game_over = self.terminal
        logic.winner = self.winner

        success, _ = logic.step(row, col, self.current_player)
        if not success:
            raise ValueError(f"Invalid action {action} for current board state")

        next_player = -self.current_player
        return GomokuState(
            size=self.size,
            board=logic.board.astype(np.int8, copy=False),
            current_player=next_player,
            last_move=(row, col),
            winner=logic.winner,
            terminal=logic.game_over,
        )

    def outcome_for(self, player: int) -> float:
        if not self.terminal or self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def is_empty(self) -> bool:
        return not np.any(self.board)

    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.board))


def random_opening_actions(state: GomokuState, rng: np.random.Generator, moves: int) -> Iterable[int]:
    available = state.legal_actions().tolist()
    if not available:
        return []
    move_count = min(moves, len(available))
    return rng.choice(available, size=move_count, replace=False).tolist()