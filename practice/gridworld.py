import numpy as np

class GridWorld:
    def __init__(self):
        # Grid dimensions (Rows, Cols)
        self.height = 3
        self.width = 4
        
        # Grid Layout
        # (Row, Col): 'Type'
        self.grid_layout = {
            (0, 3): 'GOAL',
            (1, 3): 'PIT',
            (1, 1): 'WALL'
        }
        
        # Starting position
        self.start_state = (2, 0)
        self.current_state = self.start_state
        
        # Action mapping
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.actions = [0, 1, 2, 3]
        self.action_vectors = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }

    def reset(self):
        """
        Resets the agent to the starting position.
        Returns: initial state
        """
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        Takes a step in the environment.
        Args:
            action (int): The action index (0-3).
            
        Returns:
            next_state (tuple): (row, col)
            reward (float): The reward received
            done (bool): Whether the episode has ended
        """
        # 1. Calculate tentative new position
        row, col = self.current_state
        d_row, d_col = self.action_vectors[action]
        
        new_row = row + d_row
        new_col = col + d_col
        
        # 2. Check for Boundaries (Bounce back if out of bounds)
        if new_row < 0 or new_row >= self.height or \
           new_col < 0 or new_col >= self.width:
            new_row, new_col = row, col
            
        # 3. Check for Walls (Bounce back if hitting a wall)
        if (new_row, new_col) in self.grid_layout and \
           self.grid_layout[(new_row, new_col)] == 'WALL':
            new_row, new_col = row, col
            
        # 4. Update the state
        self.current_state = (new_row, new_col)
        
        # 5. Check for Terminal States (Goal or Pit)
        reward = -0.1  # Default step cost
        done = False
        
        state_type = self.grid_layout.get(self.current_state)
        
        if state_type == 'GOAL':
            reward = 1.0
            done = True
        elif state_type == 'PIT':
            reward = -1.0
            done = True
            
        return self.current_state, reward, done
    
    def peek(self, state, action):
        """
        Simulates a step from a specific state without moving the actual agent.
        
        Args:
            state (tuple): The starting state (row, col)
            action (int): The action to take
            
        Returns:
            next_state (tuple): The resulting state
            reward (float): The reward for this transition
            done (bool): Whether this leads to a terminal state
        """
        row, col = state
        d_row, d_col = self.action_vectors[action]
        
        # 1. Calculate tentative new position
        new_row = row + d_row
        new_col = col + d_col
        
        # 2. Check Boundaries
        if new_row < 0 or new_row >= self.height or \
           new_col < 0 or new_col >= self.width:
            new_row, new_col = row, col
            
        # 3. Check Walls
        if (new_row, new_col) in self.grid_layout and \
           self.grid_layout[(new_row, new_col)] == 'WALL':
            new_row, new_col = row, col
            
        # 4. Determine Reward and Done (Goal/Pit check)
        # Note: We do NOT update self.current_state here
        
        next_state = (new_row, new_col)
        reward = -0.1  # Default step cost
        done = False
        
        state_type = self.grid_layout.get(next_state)
        
        if state_type == 'GOAL':
            reward = 1.0
            done = True
        elif state_type == 'PIT':
            reward = -1.0
            done = True
            
        return next_state, reward, done

    def render(self):
        """
        Visualizes the grid in the console.
        """
        grid = np.full((self.height, self.width), '.', dtype=str)
        
        # Place static elements
        for (r, c), type_ in self.grid_layout.items():
            if type_ == 'GOAL': grid[r, c] = 'G'
            elif type_ == 'PIT': grid[r, c] = 'P'
            elif type_ == 'WALL': grid[r, c] = 'W'
            
        # Place agent
        r, c = self.current_state
        # If agent is on top of Goal/Pit, show Agent 'A'
        grid[r, c] = 'A'
        
        print("\n".join(" ".join(row) for row in grid))
        print()

env = GridWorld()
state = env.reset()
done = False
while not done:
    action = np.random.choice([0, 1, 2, 3])
    state, reward, done = env.step(action)
    print(f"Moved to {state}, Reward: {reward}")