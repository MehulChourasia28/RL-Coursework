import random
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

from gameboard import GomokuGame
from gomoku_config import BOARD_SIZE
from Ashrayas_agent.train_dqn_gomoku import DQN


class LegacyDQN(torch.nn.Module):
    """Backward-compatible MLP used by older checkpoints."""

    def __init__(self, board_size=BOARD_SIZE, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        input_dim = board_size * board_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(__file__).resolve().parent / "Ashrayas_agent"
MODEL_PATHS = [
    MODEL_DIR / "dqn_gomoku_best.pt",
    MODEL_DIR / "dqn_gomoku.pt",
]

_model_bundle = None


def _load_model_bundle():
    for model_path in MODEL_PATHS:
        if not model_path.exists():
            continue

        checkpoint = torch.load(model_path, map_location=DEVICE)
        model_state_dict = checkpoint["model_state_dict"]
        model_board_size = int(checkpoint.get("board_size", BOARD_SIZE))
        action_dim = int(checkpoint.get("action_dim", model_board_size * model_board_size))
        model_arch = checkpoint.get("model_arch", "auto")

        if model_board_size != BOARD_SIZE:
            raise ValueError(
                f"Model board size {model_board_size} does not match configured BOARD_SIZE {BOARD_SIZE}. "
                "Retrain the model for the current board size."
            )

        is_cnn_state = any(k.startswith("encoder.") or k.startswith("head.") for k in model_state_dict.keys())
        is_mlp_state = any(k.startswith("net.") for k in model_state_dict.keys())

        if model_arch == "cnn" or (model_arch == "auto" and is_cnn_state):
            policy_net = DQN(board_size=model_board_size, action_dim=action_dim).to(DEVICE)
            selected_arch = "cnn"
        elif model_arch == "mlp" or (model_arch == "auto" and is_mlp_state):
            policy_net = LegacyDQN(board_size=model_board_size, action_dim=action_dim).to(DEVICE)
            selected_arch = "mlp"
        else:
            raise ValueError("Unsupported checkpoint format: could not infer model architecture.")

        policy_net.load_state_dict(model_state_dict)
        policy_net.eval()

        return {
            "policy": policy_net,
            "board_size": model_board_size,
            "action_dim": action_dim,
            "path": str(model_path),
            "model_arch": selected_arch,
        }

    return None


try:
    _model_bundle = _load_model_bundle()
    if _model_bundle is not None:
        print(f"Loaded DQN model from: {_model_bundle['path']}")
    else:
        print("Warning: no DQN checkpoint found. Falling back to random policy.")
except Exception as exc:
    print(f"Warning: could not load DQN model: {exc}")
    _model_bundle = None


def predict(board_state):
    """
    Your AI hook.

    Args:
        board_state (np.ndarray): board where
            0 = empty, 1 = black, -1 = white.

    Returns:
        tuple[int, int]: (x, y) move where
            x = column index (0..size-1),
            y = row index (0..size-1).
    """
    flat = board_state.ravel()
    valid_actions = np.flatnonzero(flat == 0)
    if len(valid_actions) == 0:
        return -1, -1

    if _model_bundle is None:
        row, col = random.choice(np.argwhere(board_state == 0))
        return int(col), int(row)

    with torch.no_grad():
        board_size = board_state.shape[0]
        if _model_bundle.get("model_arch") == "mlp":
            state_t = torch.as_tensor(board_state, dtype=torch.float32, device=DEVICE).view(1, -1)
        else:
            state_t = torch.as_tensor(board_state, dtype=torch.float32, device=DEVICE).view(1, 1, board_size, board_size)

        q_values = _model_bundle["policy"](state_t).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=DEVICE)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
        action = int(valid_actions[best_valid_idx])

    row = action // board_size
    col = action % board_size
    return int(col), int(row)


class GomokuHumanVsAI(GomokuGame):
    """Play Gomoku with a human against an AI defined by predict()."""

    def __init__(self, human_player=1, ai_move_delay_ms=250, size=BOARD_SIZE):
        super().__init__(size=size)
        if human_player not in (1, -1):
            raise ValueError("human_player must be 1 (black) or -1 (white)")

        self.human_player = human_player
        self.ai_player = -human_player
        self.turn = 1  # Black always starts
        self.ai_move_delay_ms = ai_move_delay_ms
        self.next_ai_move_at = None

        human_color = "Black" if self.human_player == 1 else "White"
        pygame.display.set_caption(f"Gomoku Human ({human_color}) vs AI")

    def _fallback_random_move(self):
        empty_cells = list(zip(*np.where(self.game.board == 0)))
        if not empty_cells:
            return None, None
        return random.choice(empty_cells)

    def _safe_ai_move(self):
        board_copy = self.game.board.copy()

        # The DQN is trained from Black's perspective (+1), so flip signs when AI is White.
        if self.ai_player == -1:
            board_copy = -board_copy

        try:
            x, y = predict(board_copy)
            row, col = int(y), int(x)
        except Exception as exc:
            print(f"AI predict() raised an exception: {exc}")
            row, col = self._fallback_random_move()

        if row is None:
            return

        if not self.game.is_valid_move(row, col):
            print(f"AI returned invalid move ({col}, {row}). Using random legal fallback.")
            row, col = self._fallback_random_move()
            if row is None:
                return

        success, msg = self.game.step(row, col, self.ai_player)

        if success and msg == "Continue":
            self.turn = self.human_player

    def _draw_hover_for_human(self):
        mx, my = pygame.mouse.get_pos()
        col = round((mx - self.OFFSET) / self.CELL_SIZE)
        row = round((my - self.OFFSET) / self.CELL_SIZE)

        if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
            if self.game.board[row][col] == 0:
                x = self.OFFSET + col * self.CELL_SIZE
                y = self.OFFSET + row * self.CELL_SIZE
                pygame.draw.circle(self.screen, self.HOVER_COLOR, (x, y), self.CELL_SIZE // 2 - 5, 2)

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            self.draw_board()

            restart_btn = None
            quit_btn = None

            if not self.game.game_over and self.turn == self.human_player:
                self._draw_hover_for_human()

            if self.game.game_over:
                restart_btn, quit_btn = self.draw_game_over(self.game.winner)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game.game_over:
                        if restart_btn and restart_btn.collidepoint(event.pos):
                            self.game.reset()
                            self.turn = 1
                            self.next_ai_move_at = None
                        elif quit_btn and quit_btn.collidepoint(event.pos):
                            running = False
                    else:
                        if self.turn != self.human_player:
                            continue

                        mx, my = pygame.mouse.get_pos()
                        col = round((mx - self.OFFSET) / self.CELL_SIZE)
                        row = round((my - self.OFFSET) / self.CELL_SIZE)

                        success, msg = self.game.step(row, col, self.human_player)
                        if success and msg == "Continue":
                            self.turn = self.ai_player
                            self.next_ai_move_at = pygame.time.get_ticks() + self.ai_move_delay_ms

            if not self.game.game_over and self.turn == self.ai_player:
                if self.next_ai_move_at is None:
                    self.next_ai_move_at = pygame.time.get_ticks() + self.ai_move_delay_ms

                if pygame.time.get_ticks() >= self.next_ai_move_at:
                    self.next_ai_move_at = None
                    self._safe_ai_move()

            clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # Change to -1 if you want to play White and let AI start as Black.
    game = GomokuHumanVsAI(human_player=1, ai_move_delay_ms=250, size=BOARD_SIZE)
    game.run()
