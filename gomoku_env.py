"""
  This file is a standardised wrapper around the actual gomoku game made in the gameboard.py file.
  It is desinged to create the interface for the RL algorithm to work in.
  
  During each episode, the agent selects actions (board positions 0-80) to place black stones; 
  the environment validates the move (if the move is invalid such as when it is placed ontop of an existing block then
  the game will end with a -10 reward, the idea is that the RL algorithm will learn to not do this)
  and checks for win, then immediately makes the opponent place a random white stone and checks for an opponent win. 
  This process repeats until a win occurs. There is also multiple episodes implemented in the code.
  
  The environment returns a new board state, a reward signal (-10 for invalid moves or losses,
  +10 for wins, 0 for draws, -0.1 per step).
  
"""

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
            return self.logic.board.copy(), -10, True, {"result": "Invalid"}
            
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
            import time
            pygame.event.pump() 
            pygame.display.flip()
            time.sleep(0.2)  # Small delay to see moves clearly

    def close(self):
        if self.render_mode == 'human':
            import pygame
            pygame.quit()
            
if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    num_episodes = 5  # Set how many games to play
    visualize = True  # Set to True to see the PyGame board, False to disable
    # ===================================
    
    # 1. Initialize Environment
    render_mode = 'human' if visualize else None
    env = GomokuEnv(size=9, render_mode=render_mode)
    
    print(f"\n--- Starting {num_episodes} Game(s) ---\n")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # 2. Pick a random action (simulating an untrained agent)
            action = np.random.randint(0, 81)
            
            # 3. Step the environment
            next_state, reward, done, info = env.step(action)
            
            # 4. Render if visualization is enabled
            if visualize:
                env.render()
            
            # 5. Print logs
            if reward == -10:
                print(f"  Step {step_count} - Invalid move | Action: {action} | Reward: {reward}")
            elif reward != -0.1:  # Only print noteworthy events
                print(f"  Step {step_count} - Action: {action} | Reward: {reward} | Result: {info['result']}")
            
            step_count += 1
        
        print(f"  Episode Complete! Result: {info['result']}\n")
    
    # Clean up
    env.close()
    print("All games finished!")