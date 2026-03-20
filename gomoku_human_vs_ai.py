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
    def __init__(self, board_size=BOARD_SIZE, action_dim=None, input_channels=1):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        self.board_size = board_size
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
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


class ResidualPolicyValueBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class ResidualPolicyValueNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, channels=64, num_res_blocks=4, input_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[ResidualPolicyValueBlock(channels) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.trunk(h)

        policy = self.policy_conv(h)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = self.value_conv(h)
        value = value.view(value.size(0), -1)
        value = self.value_fc(value)

        return policy_logits, value


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


def build_model_input(board_state, input_channels, device, last_move=-1):
    if input_channels <= 1:
        return board_to_cnn_tensor(board_state, device)

    planes = build_state_planes(board_state, current_player=1, last_move=last_move, device=device)
    if input_channels == 4:
        return planes
    if input_channels == 3:
        return planes[:, :3, :, :]
    if input_channels == 2:
        return planes[:, :2, :, :]

    # For unexpected channel counts, pad with zeros beyond known planes.
    if input_channels > 4:
        size = board_state.shape[0]
        padded = torch.zeros((1, input_channels, size, size), dtype=torch.float32, device=device)
        padded[:, :4, :, :] = planes
        return padded

    return board_to_cnn_tensor(board_state, device)


def _extract_action_logits(model_output):
    if torch.is_tensor(model_output):
        return model_output

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if torch.is_tensor(item):
                return item

    if isinstance(model_output, dict):
        for key in ("q_values", "logits", "policy", "policy_logits"):
            tensor = model_output.get(key)
            if torch.is_tensor(tensor):
                return tensor

    return None


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
            input_channels = int(model_bundle.get("input_channels", 1))
            state_t = build_model_input(board_state, input_channels, device, last_move=last_move)

        model_output = policy_net(state_t)
        q_values = _extract_action_logits(model_output)
        if q_values is None:
            return int(np.random.choice(valid_actions))

        q_values = q_values.squeeze(0)
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
        #workspace_root / "checkpoint.pt",
        #workspace_root / "dqn_gomoku_selfplay_good.pt",
        workspace_root / "dqn_gomoku_unified.pt",

        
    ]

    return next((p for p in candidates if p.exists()), None)


def _infer_architecture(checkpoint):
    model_arch = checkpoint.get("model_arch", "auto")
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        return None
    keys = state_dict.keys()

    has_res_dueling = any(k.startswith("input_conv.") for k in keys) and any(k.startswith("value_stream.") for k in keys)
    has_cnn = any(k.startswith("encoder.") for k in keys) and any(k.startswith("head.") for k in keys)
    has_mlp = any(k.startswith("net.") for k in keys)
    has_policy_value = any(k.startswith("stem.") for k in keys) and any(k.startswith("policy_fc.") for k in keys)

    if model_arch in {"res_dueling", "cnn", "mlp", "policy_value_resnet"}:
        return model_arch

    if has_res_dueling:
        return "res_dueling"
    if has_cnn:
        return "cnn"
    if has_mlp:
        return "mlp"
    if has_policy_value:
        return "policy_value_resnet"
    return None


def _infer_input_channels(checkpoint, model_arch):
    stored = checkpoint.get("input_channels")
    if stored is not None:
        return int(stored)

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        return None
    if model_arch == "res_dueling":
        weight = state_dict.get("input_conv.0.weight")
        if weight is not None:
            return int(weight.shape[1])
        return 4

    if model_arch == "cnn":
        weight = state_dict.get("encoder.0.weight")
        if weight is not None:
            return int(weight.shape[1])
        return 1

    if model_arch == "policy_value_resnet":
        weight = state_dict.get("stem.0.weight")
        if weight is not None:
            return int(weight.shape[1])
        return 1

    return None


def _looks_like_state_dict(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    return any(torch.is_tensor(v) for v in obj.values())


def _normalize_checkpoint_payload(loaded_obj):
    if isinstance(loaded_obj, nn.Module):
        return {
            "meta": {},
            "model_state_dict": loaded_obj.state_dict(),
            "loaded_model": loaded_obj,
        }

    if _looks_like_state_dict(loaded_obj):
        return {
            "meta": {},
            "model_state_dict": loaded_obj,
            "loaded_model": None,
        }

    if not isinstance(loaded_obj, dict):
        return None

    state_dict_candidates = [
        "model_state_dict",
        "state_dict",
        "policy_state_dict",
        "policy_net_state_dict",
        "net_state_dict",
        "q_network_state_dict",
        "weights",
        "net",
    ]
    model_candidates = ["model", "policy", "network", "net", "q_network"]

    state_dict = None
    loaded_model = None

    for key in state_dict_candidates:
        candidate = loaded_obj.get(key)
        if _looks_like_state_dict(candidate):
            state_dict = candidate
            break

    for key in model_candidates:
        candidate = loaded_obj.get(key)
        if isinstance(candidate, nn.Module):
            loaded_model = candidate
            if state_dict is None:
                state_dict = candidate.state_dict()
            break

    if state_dict is None and _looks_like_state_dict(loaded_obj):
        state_dict = loaded_obj

    if state_dict is None:
        return None

    normalized = {
        "meta": loaded_obj,
        "model_state_dict": state_dict,
        "loaded_model": loaded_model,
    }
    return normalized


def load_checkpoint(device, checkpoint_path=None):
    ckpt_path = _resolve_checkpoint_path(checkpoint_path)

    if ckpt_path is None:
        if checkpoint_path:
            print(f"Checkpoint not found: {checkpoint_path}. AI will play random moves.")
        else:
            print("No checkpoint found. AI will play random moves.")
        return None, None

    try:
        loaded_obj = torch.load(ckpt_path, map_location=device)
        normalized = _normalize_checkpoint_payload(loaded_obj)
        if normalized is None:
            print(f"Unsupported checkpoint payload in {ckpt_path}. AI will play random moves.")
            return None, None

        meta = normalized["meta"] if isinstance(normalized.get("meta"), dict) else {}
        checkpoint = {
            "model_state_dict": normalized["model_state_dict"],
            "model_arch": meta.get("model_arch", "auto"),
            "input_channels": meta.get("input_channels"),
        }

        config = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
        board_size = int(meta.get("board_size", config.get("board_size", BOARD_SIZE)))
        action_dim = int(meta.get("action_dim", board_size * board_size))
        model_arch = _infer_architecture(checkpoint)
        input_channels = _infer_input_channels(checkpoint, model_arch)
        loaded_model = normalized.get("loaded_model")
        channels = int(meta.get("channels", config.get("channels", 64)))
        num_res_blocks = int(meta.get("num_res_blocks", config.get("num_res_blocks", 4)))

        if board_size != BOARD_SIZE:
            print(
                f"Checkpoint board size ({board_size}) does not match gomoku_config.BOARD_SIZE ({BOARD_SIZE}). "
                "Model disabled. Retrain with matching board size."
            )
            return None, None

        if model_arch is None and loaded_model is not None:
            # Fallback: infer from a serialized module if metadata was not saved.
            if isinstance(loaded_model, ResidualDuelingDQN):
                model_arch = "res_dueling"
            elif isinstance(loaded_model, CnnDQN):
                model_arch = "cnn"
            elif isinstance(loaded_model, MlpDQN):
                model_arch = "mlp"
            elif isinstance(loaded_model, ResidualPolicyValueNet):
                model_arch = "policy_value_resnet"

        if model_arch is None:
            print(f"Unsupported checkpoint architecture in {ckpt_path}. AI will play random moves.")
            return None, None

        if loaded_model is not None and isinstance(loaded_model, (ResidualDuelingDQN, CnnDQN, MlpDQN, ResidualPolicyValueNet)):
            model = loaded_model.to(device)
            if input_channels is None and hasattr(model, "input_channels"):
                input_channels = int(getattr(model, "input_channels"))
        else:
            if model_arch == "res_dueling":
                model = ResidualDuelingDQN(
                    board_size=board_size,
                    action_dim=action_dim,
                    input_channels=input_channels if input_channels is not None else 4,
                ).to(device)
            elif model_arch == "mlp":
                model = MlpDQN(board_size=board_size, action_dim=action_dim).to(device)
            elif model_arch == "policy_value_resnet":
                model = ResidualPolicyValueNet(
                    board_size=board_size,
                    channels=channels,
                    num_res_blocks=num_res_blocks,
                    input_channels=input_channels if input_channels is not None else 1,
                ).to(device)
            else:
                model = CnnDQN(
                    board_size=board_size,
                    action_dim=action_dim,
                    input_channels=input_channels if input_channels is not None else 1,
                ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        input_info = f" | input_channels: {input_channels}" if input_channels is not None else ""
        print(f"Loaded model: {ckpt_path} | architecture: {model_arch}{input_info}")
        return {
            "model": model,
            "model_arch": model_arch,
            "input_channels": input_channels,
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
