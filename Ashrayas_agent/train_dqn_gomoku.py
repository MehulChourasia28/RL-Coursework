import random

import sys
from pathlib import Path

# 1. Find the parent directory of the current file
parent_dir = str(Path(__file__).resolve().parent.parent)

# 2. Add that directory to Python's search path
sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gomoku_env import GomokuEnv
from gomoku_config import BOARD_SIZE


class ResBlock(nn.Module):
    """Pre-activation residual block (no batch norm — avoids train/eval mode issues in RL)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class DQN(nn.Module):
    """Dueling DQN with residual blocks.

    Separates state-value V(s) from per-action advantage A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A).  This helps the network learn
    which *states* are good (value) independently of which *action*
    is best (advantage), dramatically improving learning efficiency.
    """
    def __init__(self, board_size=BOARD_SIZE, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        self.board_size = board_size
        channels = 64
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
        )
        # Value stream — "how good is this board position?"
        self.value_stream = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream — "how much better is action a than average?"
        self.advantage_stream = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    


class PrioritizedReplayBuffer:
    """Proportional-priority replay (Schaul et al., 2015).

    Terminal transitions (wins/losses) receive high initial priority so the
    network is never starved of reward signal, preventing catastrophic forgetting.
    """

    def __init__(self, capacity=100_000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self._pos] = (state, action, reward, next_state, done)
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        raw = self._priorities[:n] ** self.alpha
        probs = raw / raw.sum()
        # Re-normalise to guard against floating-point drift
        probs = probs / probs.sum()
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        # Importance-sampling weights to correct for the non-uniform sampling bias
        weights = (n * probs[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self._priorities[idx] = float(abs(err)) + 1e-6
        n = len(self.buffer)
        if n > 0:
            self._max_priority = float(self._priorities[:n].max())

    def __len__(self):
        return len(self.buffer)


def state_to_tensor(state, device):
    board_size = state.shape[0]
    t = torch.as_tensor(state, dtype=torch.float32, device=device)
    ch0 = (t == 1.0).float()   # agent stones
    ch1 = (t == -1.0).float()  # opponent stones
    return torch.stack([ch0, ch1], dim=0).unsqueeze(0)  # (1, 2, H, W)


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


def compute_next_q_max(policy_net, target_net, next_states, dones, device):
    with torch.no_grad():
        # Empty cells: both agent and opponent channels are 0
        valid_actions_mask = next_states.sum(dim=1).view(next_states.size(0), -1).eq(0.0)

        # Double DQN: policy_net selects the action, target_net evaluates it
        policy_q = policy_net(next_states)
        policy_q_masked = policy_q.masked_fill(~valid_actions_mask, float("-inf"))
        best_actions = policy_q_masked.argmax(dim=1, keepdim=True)

        target_q = target_net(next_states)
        next_q_max = target_q.gather(1, best_actions).squeeze(1)

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
    gamma=0.97,
    lr=1e-4,
    batch_size=256,
    replay_capacity=300_000,
    warmup_steps=5000,
    tau=0.002,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_anneal_steps=500_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=300_000,
    train_every=2,
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
    # element-wise Huber loss — will be weighted by PER importance-sampling weights
    loss_fn = nn.SmoothL1Loss(reduction="none")
    replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=per_alpha)
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
                # Anneal IS-weight correction from beta_start → 1.0 over training
                beta = min(1.0, per_beta_start + (1.0 - per_beta_start) * global_step / per_beta_anneal_steps)
                states, actions, rewards, next_states, dones, sample_indices, is_weights = replay_buffer.sample(batch_size, beta=beta)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
                is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=device)

                # Convert raw {-1,0,+1} boards to 2-channel binary tensors
                states_raw = states_t.view(batch_size, board_size, board_size)
                states_t = torch.stack([(states_raw == 1.0), (states_raw == -1.0)], dim=1).float()
                next_states_raw = next_states_t.view(batch_size, board_size, board_size)
                next_states_t = torch.stack([(next_states_raw == 1.0), (next_states_raw == -1.0)], dim=1).float()

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                next_q_max = compute_next_q_max(policy_net, target_net, next_states_t, dones_t, device)
                q_target = rewards_t + gamma * next_q_max

                # Update PER priorities with the current TD errors (before the gradient step)
                td_errors = (q_pred.detach() - q_target).abs().cpu().numpy()
                replay_buffer.update_priorities(sample_indices, td_errors)

                # IS-weighted Huber loss: high-priority samples don't dominate gradients
                loss = (is_weights_t * loss_fn(q_pred, q_target)).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

                # Soft Polyak target update — far more stable than periodic hard copies
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * pp.data)

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
