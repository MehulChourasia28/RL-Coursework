import random
import sys

import numpy as np
import pygame

from Mehuls_agent.inference import predict_move
from gameboard import GomokuGame
from gomoku_config import BOARD_SIZE


def predict(board_state, player=-1):
    """
    AI move hook backed by Mehuls_agent.

    Args:
        board_state (np.ndarray): board where
            0 = empty, 1 = black, -1 = white.
        player (int): side to move for the AI.

    Returns:
        tuple[int, int]: (x, y) move where
            x = column index (0..size-1),
            y = row index (0..size-1).
    """
    return predict_move(board_state, player)


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
        try:
            x, y = predict(board_copy, self.ai_player)
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
