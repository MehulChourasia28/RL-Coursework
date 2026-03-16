"""
Mehuls_agent — DDQN-based Gomoku agent.

Public API
----------
predict_move(board_state, player)  →  (x, y)
load_agent()                       →  DDQNAgent | None
"""

from .inference import predict_move, load_agent

__all__ = ["predict_move", "load_agent"]
