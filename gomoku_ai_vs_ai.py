import argparse
from pathlib import Path

import numpy as np
import torch

from gameboard import GomokuLogic
from gomoku_config import BOARD_SIZE
from gomoku_human_vs_ai import load_checkpoint, pick_ai_action


# Manual defaults you can edit directly in this file.
BLACK_CHECKPOINT_PATH = "dqn_gomoku_unified.pt"
WHITE_CHECKPOINT_PATH = "checkpoint.pt"
DEFAULT_GAMES = 100


def _safe_name(path_str):
    if not path_str:
        return "random"
    return Path(path_str).name


def _random_valid_action(board):
    valid = np.flatnonzero(board.ravel() == 0)
    if len(valid) == 0:
        return None
    return int(np.random.choice(valid))


def play_one_game(black_bundle, white_bundle, device):
    logic = GomokuLogic(size=BOARD_SIZE)
    turn = 1
    last_move = -1
    moves = 0

    while not logic.game_over:
        board = logic.board.copy()
        bundle = black_bundle if turn == 1 else white_bundle

        action = pick_ai_action(bundle, board, ai_player=turn, device=device, last_move=last_move)
        if action is None:
            logic.game_over = True
            logic.winner = 0
            break

        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        success, msg = logic.step(row, col, turn)

        if not success:
            fallback = _random_valid_action(logic.board)
            if fallback is None:
                logic.game_over = True
                logic.winner = 0
                break
            row = fallback // BOARD_SIZE
            col = fallback % BOARD_SIZE
            success, msg = logic.step(row, col, turn)
            action = fallback

        if success:
            last_move = action
            moves += 1

        if success and msg not in {"Win", "Draw"}:
            turn *= -1

    return logic.winner, moves


def run_match_series(black_ckpt, white_ckpt, games):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    black_bundle, black_path = load_checkpoint(device, checkpoint_path=black_ckpt)
    white_bundle, white_path = load_checkpoint(device, checkpoint_path=white_ckpt)

    black_name = _safe_name(str(black_path) if black_path else black_ckpt)
    white_name = _safe_name(str(white_path) if white_path else white_ckpt)

    black_wins = 0
    white_wins = 0
    draws = 0

    print("\n=== AI vs AI Tournament ===")
    print(f"Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Black AI: {black_name}")
    print(f"White AI: {white_name}")
    print(f"Games: {games}\n")

    for i in range(1, games + 1):
        winner, moves = play_one_game(black_bundle, white_bundle, device)

        if winner == 1:
            black_wins += 1
            result = "Black wins"
        elif winner == -1:
            white_wins += 1
            result = "White wins"
        else:
            draws += 1
            result = "Draw"

        print(f"Game {i}/{games}: {result} (moves={moves})")

    print("\n=== Final Results ===")
    print(f"Black wins: {black_wins}")
    print(f"White wins: {white_wins}")
    print(f"Draws: {draws}")
    print(f"Black win rate: {black_wins / games:.2%}")
    print(f"White win rate: {white_wins / games:.2%}")



def parse_args():
    parser = argparse.ArgumentParser(description="Run AI vs AI Gomoku games between two .pt checkpoints")
    parser.add_argument(
        "--black-checkpoint",
        type=str,
        default=None,
        help="Path to .pt model for Black (+1).",
    )
    parser.add_argument(
        "--white-checkpoint",
        type=str,
        default=None,
        help="Path to .pt model for White (-1).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help="Number of games to play (default: 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be a positive integer")

    black_ckpt = args.black_checkpoint or BLACK_CHECKPOINT_PATH
    white_ckpt = args.white_checkpoint or WHITE_CHECKPOINT_PATH

    if not black_ckpt or not white_ckpt:
        raise ValueError("Set BLACK_CHECKPOINT_PATH and WHITE_CHECKPOINT_PATH in gomoku_ai_vs_ai.py or pass --black-checkpoint and --white-checkpoint.")

    run_match_series(
        black_ckpt=black_ckpt,
        white_ckpt=white_ckpt,
        games=args.games,
    )
