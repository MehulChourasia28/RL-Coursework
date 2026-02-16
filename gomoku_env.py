import numpy as np
import random
# Import your existing logic class
from gameboard import GomokuLogic 

class GomokuEnv:
    def __init__(self, size=9, render_mode=None):
        self.size = size
        self.logic = GomokuLogic(size=size)
        self.action_space_n = size * size  # 81 actions for 9x9
        self.render_mode = render_mode
        
        # If we want to render, we initialize the UI
        if self.render_mode == 'human':
            from gameboard import GomokuGame # Import only if needed
            self.ui = GomokuGame()
        
    def reset(self):
        """Resets the board and returns the initial state."""
        self.logic.reset()
        # State is the board: 1=AI, -1=Opponent, 0=Empty
        return self.logic.board.copy()

    def step(self, action):
        """
        1. AI (Black) makes a move.
        2. Check if AI won.
        3. If not, Opponent (White) makes a random move immediately.
        4. Check if Opponent won.
        5. Return new state, reward, done, info.
        """
        # Convert scalar action (0-80) to (row, col)
        row = action // self.size
        col = action % self.size
        
        # --- 1. AI MOVE (Black / +1) ---
        valid_move = self.logic.is_valid_move(row, col)
        
        if not valid_move:
            # PENALTY: Invalid moves are bad. End game or punish heavily.
            # For simple RL, we punish heavily and end the episode so it learns valid moves faster.
            return self.logic.board.copy(), -10, False, {"result": "Invalid"}
            
        # Execute AI move
        success, msg = self.logic.step(row, col, 1) # 1 is AI
        
        if self.logic.game_over:
            if msg == "Win":
                reward = 10  # Big reward for winning
            else: # Draw
                reward = 0
            return self.logic.board.copy(), reward, True, {"result": msg}

        # --- 2. OPPONENT MOVE (White / -1) ---
        # For now, the opponent is a Random Bot. 
        # You will upgrade this to a Heuristic Bot later.
        empty_cells = list(zip(*np.where(self.logic.board == 0)))
        if not empty_cells:
            return self.logic.board.copy(), 0, True, {"result": "Draw"}
            
        opp_r, opp_c = random.choice(empty_cells)
        success, msg = self.logic.step(opp_r, opp_c, -1) # -1 is Opponent

        if self.logic.game_over:
            if msg == "Win":
                reward = -10 # Big penalty if opponent wins
                return self.logic.board.copy(), reward, True, {"result": "Loss"}
            else:
                reward = 0
                return self.logic.board.copy(), reward, True, {"result": "Draw"}

        # --- 3. GAME CONTINUES ---
        # Small penalty to encourage faster winning (optional)
        return self.logic.board.copy(), -0.1, False, {}

    def render(self):
        """
        Updates the PyGame window to show the current board state.
        """
        if self.render_mode == 'human':
            # We need to sync the Logic board with the UI game instance
            self.ui.game.board = self.logic.board
            self.ui.draw_board()
            
            # Process basic events so window doesn't freeze
            import pygame
            pygame.event.pump() 
            pygame.display.flip()

    def close(self):
        if self.render_mode == 'human':
            import pygame
            pygame.quit()
            
if __name__ == "__main__":
    # 1. Initialize Environment
    env = GomokuEnv(size=9, render_mode=None) # Set to 'human' to see PyGame window
    
    state = env.reset()
    done = False
    
    print("--- Starting Random Test Game ---")
    while not done:
        # 2. Pick a random action (simulating an untrained agent)
        action = np.random.randint(0, 81)
        
        # 3. Step the environment
        next_state, reward, done, info = env.step(action)
        
        # 4. Print logs
        if reward == -10:
            print(f"AI tried invalid move - Action: {action} | Reward: {reward} | Info: {info}")
        else:
            print(f"Action: {action} | Reward: {reward} | Info: {info}")
            
    print("Game Over!")