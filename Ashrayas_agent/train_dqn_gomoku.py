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
        self.board_size = board_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
    


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
    board_size = state.shape[0]
    return torch.as_tensor(state, dtype=torch.float32, device=device).view(1, 1, board_size, board_size)


def get_valid_actions(state):
    return np.flatnonzero(state.ravel() == 0)


def select_action(policy_net, state, epsilon, device):
    valid_actions = get_valid_actions(state)
    if len(valid_actions) == 0:
        return random.randrange(state.size)

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    with torch.no_grad():
        q_values = policy_net(state_to_tensor(state, device)).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
    return int(valid_actions[best_valid_idx])


def compute_next_q_max(target_net, next_states, dones, device):
    with torch.no_grad():
        next_q_values = target_net(next_states)
        # Mask invalid actions (occupied cells) and take the best valid action per sample.
        valid_actions_mask = next_states.squeeze(1).view(next_states.size(0), -1).eq(0.0)
        masked_next_q = next_q_values.masked_fill(~valid_actions_mask, float("-inf"))
        next_q_max = masked_next_q.max(dim=1).values

        no_valid_actions = ~valid_actions_mask.any(dim=1)
        next_q_max = torch.where(no_valid_actions, torch.zeros_like(next_q_max), next_q_max)
        return next_q_max * (1.0 - dones)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(policy_net, board_size, games, device):
    env = GomokuEnv(size=board_size, render_mode=None)
    wins = 0
    draws = 0
    losses = 0

    for _ in range(games):
        state = env.reset().astype(np.float32)
        done = False
        info = {}

        while not done:
            action = select_action(policy_net, state, epsilon=0.0, device=device)
            next_state, _, done, info = env.step(action)
            state = next_state.astype(np.float32)

        result = info.get("result", "")
        if result == "Win":
            wins += 1
        elif result == "Draw":
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def train_dqn(
    episodes=20000,
    board_size=BOARD_SIZE,
    gamma=0.99,
    lr=2.5e-4,
    batch_size=128,
    replay_capacity=300_000,
    warmup_steps=5000,
    target_update_every=1000,
    epsilon_start=1.0,
    epsilon_end=0.12,
    epsilon_decay_steps=750000,
    train_every=4,
    eval_every_episodes=500,
    eval_games=100,
    seed=42,
    save_path="dqn_gomoku.pt",
    best_save_path="dqn_gomoku_best.pt",
):
    set_global_seed(seed)
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
    min_replay_size = max(batch_size, warmup_steps)
    best_eval_win_rate = -1.0

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

            if len(replay_buffer) >= min_replay_size and global_step % train_every == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                states_t = states_t.view(batch_size, 1, board_size, board_size)
                next_states_t = next_states_t.view(batch_size, 1, board_size, board_size)

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                next_q_max = compute_next_q_max(target_net, next_states_t, dones_t, device)
                q_target = rewards_t + gamma * next_q_max

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

        if episode % eval_every_episodes == 0:
            wins, draws, losses = evaluate_policy(policy_net, board_size, eval_games, device)
            win_rate = wins / max(1, eval_games)
            print(
                f"[Eval @ ep {episode}] W/D/L: {wins}/{draws}/{losses} "
                f"| Win rate: {win_rate:.3f}"
            )

            if win_rate > best_eval_win_rate:
                best_eval_win_rate = win_rate
                torch.save(
                    {
                        "model_state_dict": policy_net.state_dict(),
                        "model_arch": "cnn",
                        "board_size": board_size,
                        "action_dim": action_dim,
                        "best_eval_win_rate": best_eval_win_rate,
                        "episode": episode,
                        "seed": seed,
                    },
                    best_save_path,
                )
                print(f"New best model saved to: {best_save_path}")

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "model_arch": "cnn",
            "board_size": board_size,
            "action_dim": action_dim,
            "best_eval_win_rate": best_eval_win_rate,
            "seed": seed,
        },
        save_path,
    )
    print(f"Training complete. Latest model saved to: {save_path}")
    if best_eval_win_rate >= 0.0:
        print(f"Best eval win rate: {best_eval_win_rate:.3f} | Best model path: {best_save_path}")


if __name__ == "__main__":
    train_dqn()
