"""
Evaluate AlphaZero agent (with full tree reuse) vs heuristic bot.
Plays N_GAMES total: half with agent as Black, half as White.
Prints win/draw/loss metrics at the end.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_gomuku5 import (Gomoku, AZNet, Node, mcts,
                         BOARD, DEVICE, N_SIMS as DEFAULT_SIMS)
from heuristics import heuristic_move

# ── Config ───────────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_DIR, "..", "models_az5", "az_iter0151.pt")
N_GAMES = 200
N_SIMS  = 400


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_net(path: str) -> AZNet:
    net = AZNet().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        net.load_state_dict(ckpt["model"])
    else:
        net.load_state_dict(ckpt)
    net.eval()
    return net


def board_to_heuristic(board: np.ndarray) -> np.ndarray:
    """Convert 0/1/2 board to 0/1/-1 board for heuristic_move."""
    h = board.copy().astype(np.int8)
    h[board == 2] = -1
    return h


def heuristic_player_id(gomoku_player: int) -> int:
    """Convert Gomoku player (1 or 2) to heuristic encoding (1 or -1)."""
    return 1 if gomoku_player == 1 else -1


# ── Single game ──────────────────────────────────────────────────────────────

def play_one_game(net: AZNet, agent_color: int, n_sims: int = N_SIMS) -> tuple:
    """
    Play one game: AZ agent vs heuristic bot.
    agent_color: 1 = agent plays Black, 2 = agent plays White.
    Returns (winner, n_moves).
      winner: 0 = draw, 1 = Black won, 2 = White won.
    """
    game = Gomoku()
    root = Node(prior=1.0)

    while True:
        if game.player == agent_color:
            # Agent's turn — MCTS with tree reuse
            pi = mcts(game, net, root, root_noise=False, n_sims=n_sims)
            action = int(np.argmax(pi))
            root = root.children.get(action, Node(prior=1.0))
        else:
            # Heuristic bot's turn
            h_board = board_to_heuristic(game.board)
            h_player = heuristic_player_id(game.player)
            r, c = heuristic_move(h_board, h_player)
            action = r * BOARD + c
            # Advance agent's tree to account for opponent's move
            root = root.children.get(action, Node(prior=1.0))

        game.move(action)

        done, winner = game.terminal()
        if done:
            return (winner, game.n_moves)


# ── Main evaluation ─────────────────────────────────────────────────────────

def run_eval(weights_path: str = DEFAULT_WEIGHTS,
             n_games: int = N_GAMES,
             n_sims: int = N_SIMS):
    print(f"Loading model: {weights_path}")
    net = load_net(weights_path)
    print(f"Device: {DEVICE} | Games: {n_games} | MCTS sims: {n_sims}")
    print(f"Agent plays {n_games // 2} games as Black, {n_games // 2} as White\n")

    half = n_games // 2
    # [wins, draws, losses] for agent
    as_black = [0, 0, 0]
    as_white = [0, 0, 0]

    for i in range(n_games):
        agent_color = 1 if i < half else 2
        side = "Black" if agent_color == 1 else "White"

        winner, n_moves = play_one_game(net, agent_color, n_sims)

        # Classify result for the agent
        if winner == 0:
            result = "Draw"
            idx = 1
        elif winner == agent_color:
            result = "Win"
            idx = 0
        else:
            result = "Loss"
            idx = 2

        if agent_color == 1:
            as_black[idx] += 1
        else:
            as_white[idx] += 1

        game_num = i + 1
        print(f"  Game {game_num:3d}/{n_games}  Agent={side}  →  {result:<4}  ({n_moves} moves)")

    # ── Summary ──────────────────────────────────────────────────────────
    total_wins   = as_black[0] + as_white[0]
    total_draws  = as_black[1] + as_white[1]
    total_losses = as_black[2] + as_white[2]

    print()
    print("=" * 50)
    print(f"  Agent vs Heuristic Bot — {n_games} games")
    print("=" * 50)
    print(f"  {'':12}  {'Wins':>6}  {'Draws':>6}  {'Losses':>6}")
    print(f"  {'As Black':12}  {as_black[0]:>6}  {as_black[1]:>6}  {as_black[2]:>6}")
    print(f"  {'As White':12}  {as_white[0]:>6}  {as_white[1]:>6}  {as_white[2]:>6}")
    print("-" * 50)
    print(f"  {'Total':12}  {total_wins:>6}  {total_draws:>6}  {total_losses:>6}")
    print(f"  Win rate: {total_wins / n_games * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AZ agent vs heuristic bot")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to model weights (.pt)")
    parser.add_argument("--games", type=int, default=N_GAMES,
                        help="Total number of games")
    parser.add_argument("--sims", type=int, default=N_SIMS,
                        help="MCTS simulations per move")
    args = parser.parse_args()
    run_eval(args.weights, args.games, args.sims)
