import numpy as np
from gridworld import GridWorld

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        """
        Args:
            env: The GridWorld environment instance.
            gamma: Discount factor (0 to 1).
            theta: The convergence threshold (stop when change < theta).
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # Initialize the Value Table V(s)
        self.V = np.zeros((env.height, env.width))

    def run_value_iteration(self):
        print("Starting Value Iteration...")
        
        iteration = 0
        while True:
            # Reset delta to 0 at the start of every sweep
            delta = 0
            
            # Loop over every state
            for i in range(self.env.height):
                for j in range(self.env.width):
                    state = (i, j)

                    # 1. SKIP TERMINAL STATES
                    # If we are at Goal or Pit, the value is 0 (game over).
                    # If we don't skip, the math breaks because we look for a "next state" that doesn't exist.
                    if state == (0, 3) or state == (1, 3):
                        continue
                    
                    # Store the old value to calculate delta later
                    old_v = self.V[state]
                    
                    val_actions = []
                    
                    # 2. Loop over actions (use 'a', not 'i'!)
                    for a in range(4):
                        next_state, reward, done = self.env.peek(state, a)
                        
                        # Get the value of the next state
                        # Check if next state is terminal? 
                        # If done is True, the Value of next state is 0.
                        if done:
                            v_next = 0
                        else:
                            # Unpack the tuple to use as indices
                            next_r, next_c = next_state
                            v_next = self.V[next_r, next_c]
                        
                        # Bellman Calculation
                        q_value = reward + (self.gamma * v_next)
                        val_actions.append(q_value)

                    # 3. Update V(s) to the max
                    best_value = max(val_actions)
                    self.V[state] = best_value
                    
                    # 4. Update delta
                    # We want to know the biggest change that happened across the WHOLE grid
                    delta = max(delta, abs(old_v - best_value))
            
            # 5. Check for convergence (Indentation is outside the for loops!)
            # If the biggest change in the grid is tiny, we are done.
            if delta < self.theta:
                break
            
            iteration += 1
            # Optional: Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Delta: {delta:.5f}")
            
        print(f"Converged in {iteration} iterations.")
        return self.V

    def get_optimal_policy(self):
        """
        After V is learned, derive the optimal policy (which action to take in each state).
        Returns:
            policy_grid: A 2D array where each cell contains the index of the best action.
        """
        # Initialize a policy grid (same shape as V) with -1 (to represent 'no action')
        policy_grid = np.full((self.env.height, self.env.width), -1, dtype=int)
        
        # Loop over all states again
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                
                # Skip Terminal States (Goal/Pit) and Walls
                # They don't have a 'next action'
                if state == (0, 3) or state == (1, 3) or state == (1, 1):
                    continue
                
                # --- EXACT SAME LOGIC AS VALUE ITERATION ---
                q_values = []
                for a in self.env.actions:
                    next_state, reward, done = self.env.peek(state, a)
                    
                    # If done is True, next value is 0
                    if done:
                        v_next = 0
                    else:
                        next_r, next_c = next_state
                        v_next = self.V[next_r, next_c]
                    
                    # Calculate Q-value
                    q = reward + (self.gamma * v_next)
                    q_values.append(q)
                
                # --- THE DIFFERENCE ---
                # Instead of storing the value, store the INDEX of the best action
                best_action_index = np.argmax(q_values)
                policy_grid[i, j] = best_action_index
                
        return policy_grid


def print_policy(policy_grid):
    # Map action indices to arrows
    # 0: Up (^), 1: Right (>), 2: Down (v), 3: Left (<), -1: .
    arrows = {0: '^', 1: '>', 2: 'v', 3: '<', -1: '.'}
    
    for row in policy_grid:
        line = ""
        for action_idx in row:
            line += arrows[action_idx] + " "
        print(line)

# --- TEST BLOCK ---
if __name__ == "__main__":
    env = GridWorld()
    agent = ValueIterationAgent(env)
    
    # Run the algorithm
    final_values = agent.run_value_iteration()
    
    # Display Results
    print("\nFinal Value Table:")
    print(np.round(final_values, 2))
    
    # Get and print policy
    policy = agent.get_optimal_policy()
    print("\nOptimal Policy (0=Up, 1=Right, 2=Down, 3=Left):")
    print_policy(policy)