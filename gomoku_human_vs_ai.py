import argparse
import random
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

from gameboard import GomokuGame
from gomoku_config import BOARD_SIZE


class CnnDQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        self.board_size = board_size
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


class MlpDQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        input_dim = board_size * board_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class ResidualDuelingDQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, action_dim=None, input_channels=4):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        channels = 64
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
        )
        self.value_stream = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_state_planes(board, current_player, last_move, device):
    board_size = board.shape[0]
    board_t = torch.as_tensor(board, dtype=torch.float32, device=device)
    player_t = torch.as_tensor(current_player, dtype=torch.float32, device=device)

    own = board_t.eq(player_t).float()
    opp = board_t.eq(-player_t).float()
    side_to_move_black = torch.full((board_size, board_size), 1.0 if current_player == 1 else 0.0, device=device)

    last_move_plane = torch.zeros((board_size, board_size), dtype=torch.float32, device=device)
    if last_move is not None and last_move >= 0:
        row = last_move // board_size
        col = last_move % board_size
        if 0 <= row < board_size and 0 <= col < board_size:
            last_move_plane[row, col] = 1.0

    return torch.stack([own, opp, side_to_move_black, last_move_plane], dim=0).unsqueeze(0)


def board_to_cnn_tensor(board, device):
    size = board.shape[0]
    return torch.as_tensor(board, dtype=torch.float32, device=device).view(1, 1, size, size)


def choose_action_from_state(model_bundle, board_state, device, last_move=-1):
    flat = board_state.ravel()
    valid_actions = np.flatnonzero(flat == 0)
    if len(valid_actions) == 0:
        return None

    with torch.no_grad():
        model_arch = model_bundle["model_arch"]
        policy_net = model_bundle["model"]

        if model_arch == "mlp":
            state_t = torch.as_tensor(board_state, dtype=torch.float32, device=device).view(1, -1)
        elif model_arch == "res_dueling":
            state_t = build_state_planes(board_state, current_player=1, last_move=last_move, device=device)
        else:
            state_t = board_to_cnn_tensor(board_state, device)

        q_values = policy_net(state_t).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
    return int(valid_actions[best_valid_idx])


def _resolve_checkpoint_path(user_path):
    if user_path:
        path = Path(user_path).expanduser().resolve()
        if path.exists():
            return path
        return None

    workspace_root = Path(__file__).resolve().parent
    candidates = [
        #workspace_root / "dqn_gomoku_selfplay_new.pt",
        #workspace_root / "dqn_gomoku_random_good.pt",
        workspace_root / "dqn_gomoku_selfplay_decent.pt",
    ]

    return next((p for p in candidates if p.exists()), None)


def _infer_architecture(checkpoint):
    model_arch = checkpoint.get("model_arch", "auto")
    state_dict = checkpoint["model_state_dict"]
    keys = state_dict.keys()

    has_res_dueling = any(k.startswith("input_conv.") for k in keys) and any(k.startswith("value_stream.") for k in keys)
    has_cnn = any(k.startswith("encoder.") for k in keys) and any(k.startswith("head.") for k in keys)
    has_mlp = any(k.startswith("net.") for k in keys)

    if model_arch in {"res_dueling", "cnn", "mlp"}:
        return model_arch

    if has_res_dueling:
        return "res_dueling"
    if has_cnn:
        return "cnn"
    if has_mlp:
        return "mlp"
    return None


def load_checkpoint(device, checkpoint_path=None):
    ckpt_path = _resolve_checkpoint_path(checkpoint_path)

    if ckpt_path is None:
        if checkpoint_path:
            print(f"Checkpoint not found: {checkpoint_path}. AI will play random moves.")
        else:
            print("No checkpoint found. AI will play random moves.")
        return None, None

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        board_size = int(checkpoint.get("board_size", BOARD_SIZE))
        action_dim = int(checkpoint.get("action_dim", board_size * board_size))
        model_arch = _infer_architecture(checkpoint)

        if board_size != BOARD_SIZE:
            print(
                f"Checkpoint board size ({board_size}) does not match gomoku_config.BOARD_SIZE ({BOARD_SIZE}). "
                "Model disabled. Retrain with matching board size."
            )
            return None, None

        if model_arch is None:
            print(f"Unsupported checkpoint architecture in {ckpt_path}. AI will play random moves.")
            return None, None

        if model_arch == "res_dueling":
            model = ResidualDuelingDQN(board_size=board_size, action_dim=action_dim, input_channels=4).to(device)
        elif model_arch == "mlp":
            model = MlpDQN(board_size=board_size, action_dim=action_dim).to(device)
        else:
            model = CnnDQN(board_size=board_size, action_dim=action_dim).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"Loaded model: {ckpt_path} | architecture: {model_arch}")
        return {
            "model": model,
            "model_arch": model_arch,
            "board_size": board_size,
            "action_dim": action_dim,
            "path": str(ckpt_path),
        }, ckpt_path
    except Exception as exc:
        print(f"Failed to load checkpoint {ckpt_path}: {exc}")
        print("AI will play random moves.")
        return None, None


def pick_ai_action(model_bundle, board, ai_player, device, last_move=-1):
    if model_bundle is None:
        valid = np.flatnonzero(board.ravel() == 0)
        if len(valid) == 0:
            return None
        return int(np.random.choice(valid))

    # The model is trained from Black's (+1) perspective.
    # Flip signs when AI is White so its own stones are still +1.
    state_for_ai = board if ai_player == 1 else -board
    return choose_action_from_state(model_bundle, state_for_ai, device, last_move=last_move)


def run_game(human_color, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, _ = load_checkpoint(device, checkpoint_path=checkpoint_path)

    game = GomokuGame(size=BOARD_SIZE)
    game.turn = 1
    last_move = -1

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
            action = pick_ai_action(model_bundle, game.game.board.copy(), ai_player, device, last_move=last_move)
            if action is None:
                game.game.game_over = True
                game.game.winner = 0
            else:
                row = action // BOARD_SIZE
                col = action % BOARD_SIZE
                success, msg = game.game.step(row, col, ai_player)
                if success:
                    last_move = action
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
                    if success:
                        last_move = row * BOARD_SIZE + col
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint. If omitted, the script searches common paths.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_game(args.human_color, checkpoint_path=args.checkpoint)
