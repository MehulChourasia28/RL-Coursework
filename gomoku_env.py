import numpy as np
import random
from gameboard import GomokuLogic 

class GomokuEnv:
    def __init__(self, size=9, render_mode=None):
        self.size = size
        self.logic = GomokuLogic(size=size)
        self.action_space_n = size * size  # 81 actions for 9x9
        self.observation_space_shape = (size, size)  # Board state shape
        self.render_mode = render_mode
        
        if self.render_mode == 'human':
            from gameboard import GomokuGame
            self.ui = GomokuGame()
        
    def reset(self):
        """Resets the board and returns the initial state."""
        self.logic.reset()
        return self.logic.board.copy()

    def step(self, action):
        """
        1. AI (Black) makes a move.
        2. Check if AI won.
        3. If not, Opponent (White) makes a random move immediately.
        4. Check if Opponent won.
        5. Return new state, reward, done, info.
        """
        row = action // self.size
        col = action % self.size
        
        # --- 1. AI MOVE (Black / +1) ---
        valid_move = self.logic.is_valid_move(row, col)
        
        if not valid_move:
            # PENALTY: Invalid moves end the episode
            return self.logic.board.copy(), -10, False, {"result": "Invalid"}  # Changed done to True
            
        success, msg = self.logic.step(row, col, 1)
        
        if self.logic.game_over:
            if msg == "Win":
                reward = 10
            else:  # Draw
                reward = 0
            return self.logic.board.copy(), reward, True, {"result": msg}

        # --- 2. OPPONENT MOVE (White / -1) ---
        empty_cells = list(zip(*np.where(self.logic.board == 0)))
        if not empty_cells:
            return self.logic.board.copy(), 0, True, {"result": "Draw"}
            
        opp_r, opp_c = random.choice(empty_cells)
        success, msg = self.logic.step(opp_r, opp_c, -1)

        if self.logic.game_over:
            if msg == "Win":
                reward = -10
                return self.logic.board.copy(), reward, True, {"result": "Loss"}
            else:
                reward = 0
                return self.logic.board.copy(), reward, True, {"result": "Draw"}

        # --- 3. GAME CONTINUES ---
        return self.logic.board.copy(), -0.1, False, {}

    def render(self):
        """Updates the PyGame window to show the current board state."""
        if self.render_mode == 'human':
            self.ui.game.board = self.logic.board
            self.ui.draw_board()
            import pygame
            pygame.event.pump() 
            pygame.display.flip()

    def close(self):
        if self.render_mode == 'human':
            import pygame
            pygame.quit()
            
if __name__ == "__main__":
    env = GomokuEnv(size=9, render_mode=None)
    
    state = env.reset()
    done = False
    
    print("--- Starting Random Test Game ---")
    while not done:
        action = np.random.randint(0, 81)
        next_state, reward, done, info = env.step(action)
        
        if reward == -10:
            print(f"AI tried invalid move - Action: {action} | Reward: {reward} | Info: {info}")
        else:
            print(f"Action: {action} | Reward: {reward} | Info: {info}")  # Fixed typo
            
    print("Game Over!")