import random
from collections import deque

import sys
from pathlib import Path

# 1. Find the parent directory of the current file
parent_dir = str(Path(__file__).resolve().parent.parent)

# 2. Add that directory to Python's search path
sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gomoku_env import GomokuEnv
from gomoku_config import BOARD_SIZE


class DQN(nn.Module):
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
    


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def state_to_tensor(state, device):
    return torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)


def get_valid_actions(state):
    return np.flatnonzero(state.flatten() == 0)


def select_action(policy_net, state, epsilon, device):
    valid_actions = get_valid_actions(state)
    if len(valid_actions) == 0:
        return random.randrange(state.size)

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    with torch.no_grad():
        q_values = policy_net(state_to_tensor(state, device)).squeeze(0).cpu().numpy()

    masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
    masked_q[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked_q))


def compute_next_q_max(target_net, next_states, dones, device):
    with torch.no_grad():
        next_q_values = target_net(next_states)

        max_values = []
        next_states_np = next_states.detach().cpu().numpy()
        for i in range(next_states_np.shape[0]):
            if dones[i] >= 0.5:
                max_values.append(0.0)
                continue

            valid_actions = np.flatnonzero(next_states_np[i] == 0)
            if len(valid_actions) == 0:
                max_values.append(0.0)
            else:
                max_values.append(torch.max(next_q_values[i, valid_actions]).item())

        return torch.tensor(max_values, dtype=torch.float32, device=device)


def train_dqn(
    episodes=50000,
    board_size=BOARD_SIZE,
    gamma=0.99,
    lr=2.5e-4,
    batch_size=128,
    replay_capacity=300_000,
    warmup_steps=5000,
    target_update_every=1000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=300000,
    save_path="dqn_gomoku.pt",
):
    env = GomokuEnv(size=board_size, render_mode=None)
    action_dim = env.action_space_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(board_size=board_size, action_dim=action_dim).to(device)
    target_net = DQN(board_size=board_size, action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Huber loss is typically more stable than MSE for temporal-difference targets.
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    global_step = 0

    for episode in range(1, episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        episode_reward = 0.0

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
                0.0, (epsilon_decay_steps - global_step) / epsilon_decay_steps
            )

            action = select_action(policy_net, state, epsilon, device)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)

            replay_buffer.add(state.flatten(), action, reward, next_state.flatten(), done)

            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay_buffer) >= max(batch_size, warmup_steps):
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                next_q_max = compute_next_q_max(target_net, next_states_t, dones_t, device)
                q_target = rewards_t + gamma * (1.0 - dones_t) * next_q_max

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0 or episode == 1:
            print(
                f"Episode {episode:>6}/{episodes} | "
                f"Reward: {episode_reward:>7.2f} | "
                f"Epsilon: {epsilon:>5.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "board_size": board_size,
            "action_dim": action_dim,
        },
        save_path,
    )
    print(f"Training complete. Model saved to: {save_path}")


if __name__ == "__main__":
    train_dqn()
