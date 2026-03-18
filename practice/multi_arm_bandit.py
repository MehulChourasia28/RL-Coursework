import numpy as np
import random
import matplotlib.pyplot as plt

lever_probs = np.array([0.2, 0.5, 0.1, 0.2])
action_space_size = lever_probs.size
iters = 1000
epsilon = 0.1

actions = np.array([])
rewards = np.array([])

def estimate(action):
    """Estimate the value of an action based on past rewards."""
    mask = (actions == action).astype(int)
    if np.sum(mask) == 0:
        return 0  # Return 0 for unexplored actions
    return np.dot(mask, rewards) / np.sum(mask)

def select_action():
    """Select an action using epsilon-greedy strategy."""
    if random.random() < epsilon:
        return random.randint(1, action_space_size)
    else:
        return np.argmax([estimate(a) for a in range(1, action_space_size + 1)]) + 1

def pull_lever(action):
    """Simulate pulling a lever - returns 1 (win) or 0 (lose) based on lever probability."""
    prob = lever_probs[action - 1]  # action is 1-indexed
    return 1 if random.random() < prob else 0

def run_bandit():
    """Run the multi-arm bandit simulation."""
    global actions, rewards
    
    actions = np.array([])
    rewards = np.array([])
    cumulative_rewards = []
    action_counts = np.zeros(action_space_size)
    
    for i in range(iters):
        # Select and perform action
        action = select_action()
        reward = pull_lever(action)
        
        # Store results
        actions = np.append(actions, action)
        rewards = np.append(rewards, reward)
        action_counts[action - 1] += 1
        
        # Track cumulative reward
        cumulative_rewards.append(np.sum(rewards))
    
    return cumulative_rewards, action_counts

def plot_results(cumulative_rewards, action_counts):
    """Plot the results of the bandit simulation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Cumulative rewards over time
    axes[0].plot(cumulative_rewards)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Rewards Over Time')
    
    # Plot 2: Action selection distribution
    axes[1].bar(range(1, action_space_size + 1), action_counts)
    axes[1].set_xlabel('Action (Lever)')
    axes[1].set_ylabel('Times Selected')
    axes[1].set_title('Action Selection Distribution')
    axes[1].set_xticks(range(1, action_space_size + 1))
    
    # Plot 3: Estimated vs True probabilities
    estimated_probs = [estimate(a) for a in range(1, action_space_size + 1)]
    x = np.arange(action_space_size)
    width = 0.35
    axes[2].bar(x - width/2, lever_probs, width, label='True Probability')
    axes[2].bar(x + width/2, estimated_probs, width, label='Estimated Probability')
    axes[2].set_xlabel('Action (Lever)')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('True vs Estimated Probabilities')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(range(1, action_space_size + 1))
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"Multi-Arm Bandit with {action_space_size} levers")
    print(f"True probabilities: {lever_probs}")
    print(f"Epsilon: {epsilon}, Iterations: {iters}")
    print("-" * 40)
    
    cumulative_rewards, action_counts = run_bandit()
    
    print(f"\nResults after {iters} iterations:")
    print(f"Total reward: {int(cumulative_rewards[-1])}")
    print(f"Average reward per pull: {cumulative_rewards[-1] / iters:.3f}")
    print(f"\nAction counts: {action_counts.astype(int)}")
    print(f"Best lever (highest true prob): Lever {np.argmax(lever_probs) + 1}")
    print(f"Most selected lever: Lever {np.argmax(action_counts) + 1}")
    
    # Show estimated values
    print("\nEstimated probabilities:")
    for a in range(1, action_space_size + 1):
        print(f"  Lever {a}: {estimate(a):.3f} (true: {lever_probs[a-1]:.1f})")
    
    plot_results(cumulative_rewards, action_counts)

