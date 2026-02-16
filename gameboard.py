import pygame
import numpy as np
import sys


# class for the logic of the board 
class GomokuLogic:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.game_over = False
        self.winner = 0  # 1: Black, -1: White, 0: None

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.game_over = False
        self.winner = 0

    def is_valid_move(self, row, col):
        # Check boundaries and if cell is empty
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def step(self, row, col, player):
        """
        Places a stone and checks for win.
        Returns: (success_bool, message_string)
        """
        if self.game_over: #if game over == true then return false since we dont want a step
            return False, "Game Over"

        if not self.is_valid_move(row, col):
            return False, "Invalid Move"

        # Place the stone
        self.board[row][col] = player

        # Check if this move won the game
        if self.check_win(row, col, player):
            self.game_over = True
            self.winner = player
            return True, "Win"
        
        # Check for Draw (Board Full)
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
            return True, "Draw"

        return True, "Continue"

    def check_win(self, row, col, player):
        """
        Checks 4 directions around the placed stone for 5-in-a-row.
        Directions: Horizontal, Vertical, Diagonal (\), Anti-Diagonal (/)
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # Count the stone we just placed
            
            # 1. Check in the positive direction
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 2. Check in the negative direction
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 3. Did we find 5 or more?
            if count >= 5:
                return True
                
        return False

# ==========================================
# PART 2: THE UI (Visualisation)
# ==========================================
class GomokuGame:
    def __init__(self):
        pygame.init()
        self.BOARD_SIZE = 9
        self.CELL_SIZE = 60
        self.OFFSET = self.CELL_SIZE // 2
        self.WIDTH = self.CELL_SIZE * self.BOARD_SIZE
        self.HEIGHT = self.CELL_SIZE * self.BOARD_SIZE
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(f"Gomoku {self.BOARD_SIZE}x{self.BOARD_SIZE}")

        # Colors
        self.WOOD_COLOR = (222, 184, 135)
        self.BLACK = (0, 0, 0)
        self.WHITE = (240, 240, 240)
        self.LINE_COLOR = (50, 50, 50)
        self.HOVER_COLOR = (100, 100, 100)
        
        # UI Colors
        self.OVERLAY_COLOR = (0, 0, 0, 180)  # Black with transparency
        self.PANEL_COLOR = (250, 250, 250)
        self.BUTTON_COLOR = (70, 130, 180)   # Steel Blue
        self.BUTTON_HOVER = (100, 149, 237)  # Cornflower Blue
        self.TEXT_COLOR = (50, 50, 50)

        self.game = GomokuLogic(size=self.BOARD_SIZE)
        self.turn = 1

    def draw_board(self):
        self.screen.fill(self.WOOD_COLOR)
        for i in range(self.BOARD_SIZE):
            # Horizontal & Vertical Lines
            start_h = (self.OFFSET, self.OFFSET + i * self.CELL_SIZE)
            end_h = (self.WIDTH - self.OFFSET, self.OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_h, end_h, 2)
            
            start_v = (self.OFFSET + i * self.CELL_SIZE, self.OFFSET)
            end_v = (self.OFFSET + i * self.CELL_SIZE, self.HEIGHT - self.OFFSET)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_v, end_v, 2)

        # Draw Stones
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.game.board[r][c] != 0:
                    x = self.OFFSET + c * self.CELL_SIZE
                    y = self.OFFSET + r * self.CELL_SIZE
                    color = self.BLACK if self.game.board[r][c] == 1 else self.WHITE
                    pygame.draw.circle(self.screen, color, (x, y), self.CELL_SIZE // 2 - 5)
                    # Add a subtle rim to white stones so they pop
                    if self.game.board[r][c] == -1:
                        pygame.draw.circle(self.screen, self.LINE_COLOR, (x, y), self.CELL_SIZE // 2 - 5, 1)

    def draw_game_over(self, winner):
        # 1. Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))

        # 2. Main Panel (Centered)
        panel_w, panel_h = 400, 250
        panel_x = (self.WIDTH - panel_w) // 2
        panel_y = (self.HEIGHT - panel_h) // 2
        
        # Drop shadow (offset by 4px)
        pygame.draw.rect(self.screen, (0,0,0,100), (panel_x+4, panel_y+4, panel_w, panel_h), border_radius=12)
        # Main Box
        pygame.draw.rect(self.screen, self.PANEL_COLOR, (panel_x, panel_y, panel_w, panel_h), border_radius=12)
        pygame.draw.rect(self.screen, self.LINE_COLOR, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=12)

        # 3. Text
        font_big = pygame.font.SysFont("Arial", 48, bold=True)
        font_small = pygame.font.SysFont("Arial", 24)
        
        if winner == 1:
            msg = "Black Wins!"
            color = self.BLACK
        elif winner == -1:
            msg = "White Wins!"
            color = (100, 100, 100) # Dark Grey for visibility
        else:
            msg = "It's a Draw!"
            color = self.LINE_COLOR

        text_surf = font_big.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, panel_y + 60))
        self.screen.blit(text_surf, text_rect)

        # 4. Buttons (Play Again / Quit)
        mouse_pos = pygame.mouse.get_pos()
        
        # Button Dimensions
        btn_w, btn_h = 140, 50
        btn_y = panel_y + 150
        btn1_x = panel_x + 40
        btn2_x = panel_x + panel_w - 40 - btn_w
        
        # Define Rects
        restart_rect = pygame.Rect(btn1_x, btn_y, btn_w, btn_h)
        quit_rect = pygame.Rect(btn2_x, btn_y, btn_h, btn_h) # Square-ish for Quit

        # Draw Play Again Button
        color1 = self.BUTTON_HOVER if restart_rect.collidepoint(mouse_pos) else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, color1, restart_rect, border_radius=8)
        
        txt1 = font_small.render("Play Again", True, (255,255,255))
        txt1_rect = txt1.get_rect(center=restart_rect.center)
        self.screen.blit(txt1, txt1_rect)

        # Draw Quit Button
        color2 = (200, 80, 80) if quit_rect.collidepoint(mouse_pos) else (180, 60, 60)
        pygame.draw.rect(self.screen, color2, quit_rect, border_radius=8)
        
        txt2 = font_small.render("X", True, (255,255,255))
        txt2_rect = txt2.get_rect(center=quit_rect.center)
        self.screen.blit(txt2, txt2_rect)

        return restart_rect, quit_rect

    def run(self):
        running = True
        while running:
            # Always draw the board first
            self.draw_board()

            # --- HOVER EFFECT ---
            if not self.game.game_over:
                mx, my = pygame.mouse.get_pos()
                col = round((mx - self.OFFSET) / self.CELL_SIZE)
                row = round((my - self.OFFSET) / self.CELL_SIZE)
                
                if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
                    if self.game.board[row][col] == 0:
                        x = self.OFFSET + col * self.CELL_SIZE
                        y = self.OFFSET + row * self.CELL_SIZE
                        pygame.draw.circle(self.screen, self.HOVER_COLOR, (x, y), self.CELL_SIZE // 2 - 5, 2)

            # --- GAME OVER SCREEN ---
            restart_btn = None
            quit_btn = None

            if self.game.game_over:
                # Draw the fancy screen and get button rectangles back
                restart_btn, quit_btn = self.draw_game_over(self.game.winner)

            pygame.display.flip()

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game.game_over:
                        # Handle UI Button Clicks
                        if restart_btn and restart_btn.collidepoint(event.pos):
                            self.game.reset()
                            self.turn = 1
                        elif quit_btn and quit_btn.collidepoint(event.pos):
                            running = False
                    else:
                        # Handle Game Moves
                        mx, my = pygame.mouse.get_pos()
                        col = round((mx - self.OFFSET) / self.CELL_SIZE)
                        row = round((my - self.OFFSET) / self.CELL_SIZE)
                        
                        success, msg = self.game.step(row, col, self.turn)
                        if success and msg != "Win" and msg != "Draw":
                            self.turn *= -1

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GomokuGame()
    game.run()