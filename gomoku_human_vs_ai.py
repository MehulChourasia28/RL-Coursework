import argparse
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn

from gameboard import GomokuGame
from gomoku_config import BOARD_SIZE


class DQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


def board_to_tensor(board, device):
    size = board.shape[0]
    return torch.as_tensor(board, dtype=torch.float32, device=device).view(1, 1, size, size)


def choose_action_from_state(policy_net, board_state, device):
    flat = board_state.ravel()
    valid_actions = np.flatnonzero(flat == 0)
    if len(valid_actions) == 0:
        return None

    with torch.no_grad():
        q_values = policy_net(board_to_tensor(board_state, device)).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
    return int(valid_actions[best_valid_idx])


def load_checkpoint(device):
    candidates = [
        Path(__file__).resolve().parent / "Ashrayas_agent" / "dqn_gomoku_best.pt",
        Path(__file__).resolve().parent / "Ashrayas_agent" / "dqn_gomoku.pt",
        Path(__file__).resolve().parent / "dqn_gomoku_best.pt",
        Path(__file__).resolve().parent / "dqn_gomoku.pt",
    ]

    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print("No checkpoint found. AI will play random moves.")
        return None, None

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        board_size = int(checkpoint.get("board_size", BOARD_SIZE))
        action_dim = int(checkpoint.get("action_dim", board_size * board_size))

        if board_size != BOARD_SIZE:
            print(
                f"Checkpoint board size ({board_size}) does not match gomoku_config.BOARD_SIZE ({BOARD_SIZE}). "
                "Model disabled. Retrain with matching board size."
            )
            return None, None

        model = DQN(board_size=board_size, action_dim=action_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded model: {ckpt_path}")
        return model, ckpt_path
    except Exception as exc:
        print(f"Failed to load checkpoint {ckpt_path}: {exc}")
        print("AI will play random moves.")
        return None, None


def pick_ai_action(policy_net, board, ai_player, device):
    if policy_net is None:
        valid = np.flatnonzero(board.ravel() == 0)
        if len(valid) == 0:
            return None
        return int(np.random.choice(valid))

    # The model is trained from Black's (+1) perspective.
    # Flip signs when AI is White so its own stones are still +1.
    state_for_ai = board if ai_player == 1 else -board
    return choose_action_from_state(policy_net, state_for_ai, device)


def run_game(human_color):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net, _ = load_checkpoint(device)

    game = GomokuGame(size=BOARD_SIZE)
    game.turn = 1

    human_player = 1 if human_color == "black" else -1
    ai_player = -human_player

    print(f"You are {human_color}. AI is {'white' if human_color == 'black' else 'black' }.")

    clock = pygame.time.Clock()
    running = True

    while running:
        game.draw_board()

        restart_btn = None
        quit_btn = None
        if game.game.game_over:
            restart_btn, quit_btn = game.draw_game_over(game.game.winner)

        pygame.display.flip()

        if not game.game.game_over and game.turn == ai_player:
            action = pick_ai_action(policy_net, game.game.board.copy(), ai_player, device)
            if action is None:
                game.game.game_over = True
                game.game.winner = 0
            else:
                row = action // BOARD_SIZE
                col = action % BOARD_SIZE
                success, msg = game.game.step(row, col, ai_player)
                if success and msg not in {"Win", "Draw"}:
                    game.turn *= -1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.game.game_over:
                    if restart_btn and restart_btn.collidepoint(event.pos):
                        game.game.reset()
                        game.turn = 1
                    elif quit_btn and quit_btn.collidepoint(event.pos):
                        running = False
                elif game.turn == human_player:
                    mx, my = pygame.mouse.get_pos()
                    col = round((mx - game.OFFSET) / game.CELL_SIZE)
                    row = round((my - game.OFFSET) / game.CELL_SIZE)
                    success, msg = game.game.step(row, col, human_player)
                    if success and msg not in {"Win", "Draw"}:
                        game.turn *= -1

        clock.tick(60)

    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Play Gomoku against a trained DQN agent")
    parser.add_argument(
        "--human-color",
        choices=["black", "white"],
        default="black",
        help="Choose your side: black moves first, white moves second.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_game(args.human_color)
