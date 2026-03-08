import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Make project root importable when this script runs from Ashrayas_agent/.
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

from gameboard import GomokuLogic
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
    def __init__(self, capacity=200_000):
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
        return None

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    with torch.no_grad():
        q_values = policy_net(state_to_tensor(state, device)).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
    return int(valid_actions[best_valid_idx])


def compute_next_q_max(target_net, next_states, dones):
    with torch.no_grad():
        next_q_values = target_net(next_states)
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


def load_initial_model(policy_net, checkpoint_path, device):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    checkpoint = torch.load(ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing model_state_dict.")

    ckpt_board_size = int(checkpoint.get("board_size", BOARD_SIZE))
    if ckpt_board_size != policy_net.board_size:
        raise ValueError(
            f"Checkpoint board_size={ckpt_board_size} does not match requested board_size={policy_net.board_size}."
        )

    policy_net.load_state_dict(state_dict)
    return checkpoint


def run_self_play_episode(policy_net, board_size, epsilon, step_penalty, device):
    logic = GomokuLogic(size=board_size)
    state = logic.board.copy().astype(np.float32)
    current_player = 1
    done = False
    episode_reward = 0.0
    move_count = 0
    transitions = []

    while not done:
        perspective_state = (state * current_player).astype(np.float32)
        action = select_action(policy_net, perspective_state, epsilon, device)

        if action is None:
            # No legal action means draw (safety fallback).
            reward = 0.0
            done = True
            next_perspective_state = perspective_state
            transitions.append((perspective_state.flatten(), 0, reward, next_perspective_state.flatten(), done))
            episode_reward += reward
            break

        row, col = divmod(action, board_size)
        success, msg = logic.step(row, col, current_player)

        if not success:
            reward = -10.0
            done = True
        elif msg == "Win":
            reward = 10.0
            done = True
        elif msg == "Draw":
            reward = 0.0
            done = True
        else:
            reward = step_penalty
            done = False

        state_after = logic.board.copy().astype(np.float32)
        next_player = -current_player
        next_perspective_state = (state_after * next_player).astype(np.float32)

        transitions.append(
            (perspective_state.flatten(), action, reward, next_perspective_state.flatten(), done)
        )
        episode_reward += reward
        move_count += 1

        state = state_after
        current_player = next_player

    return transitions, episode_reward, move_count


def train_self_play(
    episodes=25_000,
    board_size=BOARD_SIZE,
    gamma=0.99,
    lr=2.0e-4,
    batch_size=128,
    replay_capacity=300_000,
    warmup_steps=8_000,
    target_update_every=1_000,
    epsilon_start=0.25,
    epsilon_end=0.02,
    epsilon_decay_steps=250_000,
    train_every=4,
    step_penalty=-0.02,
    seed=42,
    init_checkpoint="Ashrayas_agent/dqn_gomoku.pt",
    save_path="Ashrayas_agent/dqn_gomoku_selfplay.pt",
):
    set_global_seed(seed)

    action_dim = board_size * board_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(board_size=board_size, action_dim=action_dim).to(device)
    init_meta = load_initial_model(policy_net, init_checkpoint, device)

    target_net = DQN(board_size=board_size, action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    min_replay_size = max(batch_size, warmup_steps)

    global_step = 0

    print(f"Initialized from: {init_checkpoint}")
    if "best_eval_win_rate" in init_meta:
        print(f"Base checkpoint eval win rate: {init_meta['best_eval_win_rate']:.3f}")

    for episode in range(1, episodes + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, (epsilon_decay_steps - global_step) / epsilon_decay_steps
        )

        transitions, episode_reward, move_count = run_self_play_episode(
            policy_net, board_size, epsilon, step_penalty, device
        )
        for transition in transitions:
            replay_buffer.add(*transition)
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
                next_q_max = compute_next_q_max(target_net, next_states_t, dones_t)
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
                f"Moves: {move_count:>3} | "
                f"Reward: {episode_reward:>7.2f} | "
                f"Epsilon: {epsilon:>5.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "model_arch": "cnn",
            "board_size": board_size,
            "action_dim": action_dim,
            "seed": seed,
            "trained_from": str(init_checkpoint),
            "training_mode": "self_play",
        },
        save_path,
    )
    print(f"Self-play training complete. Saved model to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Continue Gomoku DQN training via self-play")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=300000)
    parser.add_argument("--warmup-steps", type=int, default=8000)
    parser.add_argument("--target-update-every", type=int, default=1000)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--step-penalty", type=float, default=-0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", type=str, default="Ashrayas_agent/dqn_gomoku.pt")
    parser.add_argument("--save-path", type=str, default="Ashrayas_agent/dqn_gomoku_selfplay.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_self_play(
        episodes=args.episodes,
        board_size=args.board_size,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        target_update_every=args.target_update_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        train_every=args.train_every,
        step_penalty=args.step_penalty,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        save_path=args.save_path,
    )
