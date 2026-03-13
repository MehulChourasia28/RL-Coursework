import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Make project root importable when this script runs from Ashrayas_agent/.
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

from gameboard import GomokuLogic
from gomoku_config import BOARD_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)


class DQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, action_dim=None, input_channels=4):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        self.board_size = board_size
        channels = 64
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
        )
        self.value_stream = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
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


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        current_player,
        last_move,
        action,
        reward,
        next_state,
        next_player,
        next_last_move,
        done,
    ):
        self.buffer.append(
            (
                state,
                current_player,
                last_move,
                action,
                reward,
                next_state,
                next_player,
                next_last_move,
                done,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        (
            states,
            players,
            last_moves,
            actions,
            rewards,
            next_states,
            next_players,
            next_last_moves,
            dones,
        ) = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(players, dtype=np.int64),
            np.array(last_moves, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_players, dtype=np.int64),
            np.array(next_last_moves, dtype=np.int64),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def build_state_planes(boards, players, last_moves, board_size):
    batch_size = boards.size(0)
    players_exp = players.view(batch_size, 1, 1)

    own = boards.eq(players_exp).float()
    opp = boards.eq(-players_exp).float()
    side_to_move_black = players.eq(1).float().view(batch_size, 1, 1).expand(-1, board_size, board_size)

    last_move_plane = torch.zeros(
        (batch_size, board_size, board_size),
        dtype=torch.float32,
        device=boards.device,
    )
    valid = last_moves.ge(0)
    if valid.any():
        batch_idx = torch.arange(batch_size, device=boards.device)[valid]
        rows = torch.div(last_moves[valid], board_size, rounding_mode="floor")
        cols = torch.remainder(last_moves[valid], board_size)
        last_move_plane[batch_idx, rows, cols] = 1.0

    return torch.stack([own, opp, side_to_move_black, last_move_plane], dim=1)


def state_to_tensor(state, current_player, last_move, device):
    board_size = state.shape[0]
    board_t = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, board_size, board_size)
    player_t = torch.as_tensor([current_player], dtype=torch.int64, device=device)
    last_move_t = torch.as_tensor([last_move], dtype=torch.int64, device=device)
    return build_state_planes(board_t, player_t, last_move_t, board_size)


def get_valid_actions(state):
    return np.flatnonzero(state.ravel() == 0)


def select_action(policy_net, state, current_player, last_move, epsilon, device):
    valid_actions = get_valid_actions(state)
    if len(valid_actions) == 0:
        return None

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    with torch.no_grad():
        q_values = policy_net(state_to_tensor(state, current_player, last_move, device)).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
    return int(valid_actions[best_valid_idx])


def transform_action_index(action_idx, sym, board_size):
    row, col = divmod(int(action_idx), board_size)

    if sym == 0:  # identity
        nr, nc = row, col
    elif sym == 1:  # rot90 ccw
        nr, nc = board_size - 1 - col, row
    elif sym == 2:  # rot180
        nr, nc = board_size - 1 - row, board_size - 1 - col
    elif sym == 3:  # rot270 ccw
        nr, nc = col, board_size - 1 - row
    elif sym == 4:  # flip left-right
        nr, nc = row, board_size - 1 - col
    elif sym == 5:  # flip up-down
        nr, nc = board_size - 1 - row, col
    elif sym == 6:  # main diagonal transpose
        nr, nc = col, row
    elif sym == 7:  # anti-diagonal
        nr, nc = board_size - 1 - col, board_size - 1 - row
    else:
        raise ValueError(f"Invalid symmetry id: {sym}")

    return nr * board_size + nc


def apply_symmetry_to_tensor(x, sym):
    if sym == 0:
        return x
    if sym == 1:
        return torch.rot90(x, 1, dims=[1, 2])
    if sym == 2:
        return torch.rot90(x, 2, dims=[1, 2])
    if sym == 3:
        return torch.rot90(x, 3, dims=[1, 2])
    if sym == 4:
        return torch.flip(x, dims=[2])
    if sym == 5:
        return torch.flip(x, dims=[1])
    if sym == 6:
        return x.transpose(1, 2)
    if sym == 7:
        return torch.flip(x.transpose(1, 2), dims=[1, 2])
    raise ValueError(f"Invalid symmetry id: {sym}")


def augment_batch_symmetry(states, actions, next_states, board_size):
    batch_size = states.size(0)
    for i in range(batch_size):
        sym = random.randrange(8)
        if sym == 0:
            continue
        # Some transforms (e.g. transpose) return aliased views; clone before assignment.
        states_aug = apply_symmetry_to_tensor(states[i], sym).clone()
        next_states_aug = apply_symmetry_to_tensor(next_states[i], sym).clone()
        states[i] = states_aug
        next_states[i] = next_states_aug
        actions[i] = transform_action_index(actions[i].item(), sym, board_size)


def compute_next_q_max(policy_net, target_net, next_states, next_boards, dones):
    with torch.no_grad():
        valid_actions_mask = next_boards.view(next_boards.size(0), -1).eq(0.0)

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


def _load_state_dict_with_compat(policy_net, state_dict):
    try:
        policy_net.load_state_dict(state_dict)
        return "loaded"
    except RuntimeError:
        model_state = policy_net.state_dict()
        adapted = {}

        for key, value in state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape == value.shape:
                adapted[key] = value
                continue

            if (
                key == "input_conv.0.weight"
                and value.ndim == 4
                and model_state[key].ndim == 4
                and value.shape[2:] == model_state[key].shape[2:]
                and value.shape[0] == model_state[key].shape[0]
                and value.shape[1] == 2
                and model_state[key].shape[1] == 4
            ):
                new_w = model_state[key].clone()
                new_w[:, :2, :, :] = value
                adapted[key] = new_w

        policy_net.load_state_dict(adapted, strict=False)
        return "partially_loaded"


def load_initial_model(policy_net, checkpoint_path, device):
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        print("No init checkpoint provided. Starting self-play training from scratch.")
        return {}

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

    mode = _load_state_dict_with_compat(policy_net, state_dict)
    if mode == "partially_loaded":
        print("Checkpoint loaded with compatibility fallback (partial parameter transfer).")
    return checkpoint


def run_self_play_episode(policy_net, board_size, epsilon, step_penalty, device):
    logic = GomokuLogic(size=board_size)
    state = logic.board.copy().astype(np.float32)
    current_player = 1
    last_move = -1
    done = False
    episode_reward = 0.0
    move_count = 0
    transitions = []

    while not done:
        action = select_action(policy_net, state, current_player, last_move, epsilon, device)

        if action is None:
            # No legal action means draw (safety fallback).
            reward = 0.0
            done = True
            transitions.append(
                (
                    state.flatten(),
                    current_player,
                    last_move,
                    0,
                    reward,
                    state.flatten(),
                    current_player,
                    last_move,
                    done,
                )
            )
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
        next_last_move = action

        transitions.append(
            (
                state.flatten(),
                current_player,
                last_move,
                action,
                reward,
                state_after.flatten(),
                next_player,
                next_last_move,
                done,
            )
        )
        episode_reward += reward
        move_count += 1

        state = state_after
        current_player = next_player
        last_move = next_last_move

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
    epsilon_start=0.7,
    epsilon_end=0.1,
    epsilon_decay_steps=250_000,
    train_every=4,
    step_penalty=-0.02,
    use_symmetry_augmentation=True,
    seed=42,
    init_checkpoint=None,
    save_path="Ashrayas_agent/dqn_gomoku_selfplay2.pt",
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

    if init_checkpoint:
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
                (
                    states,
                    players,
                    last_moves,
                    actions,
                    rewards,
                    next_states,
                    next_players,
                    next_last_moves,
                    dones,
                ) = replay_buffer.sample(batch_size)

                states_board_t = torch.tensor(states, dtype=torch.float32, device=device).view(
                    batch_size, board_size, board_size
                )
                players_t = torch.tensor(players, dtype=torch.int64, device=device)
                last_moves_t = torch.tensor(last_moves, dtype=torch.int64, device=device)

                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

                next_states_board_t = torch.tensor(next_states, dtype=torch.float32, device=device).view(
                    batch_size, board_size, board_size
                )
                next_players_t = torch.tensor(next_players, dtype=torch.int64, device=device)
                next_last_moves_t = torch.tensor(next_last_moves, dtype=torch.int64, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                states_t = build_state_planes(states_board_t, players_t, last_moves_t, board_size)
                next_states_t = build_state_planes(next_states_board_t, next_players_t, next_last_moves_t, board_size)

                if use_symmetry_augmentation:
                    augment_batch_symmetry(states_t, actions_t, next_states_t, board_size)
                    states_board_t = states_t[:, 0] - states_t[:, 1]
                    next_states_board_t = next_states_t[:, 0] - next_states_t[:, 1]

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                next_q_max = compute_next_q_max(policy_net, target_net, next_states_t, next_states_board_t, dones_t)
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
            "model_arch": "residual_dueling_dqn_4plane",
            "board_size": board_size,
            "action_dim": action_dim,
            "seed": seed,
            "trained_from": str(init_checkpoint) if init_checkpoint else "scratch",
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
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint to initialize from. Leave empty to train from scratch.",
    )
    parser.add_argument(
        "--no-symmetry-augmentation",
        action="store_true",
        help="Disable random rotation/flip augmentation during replay sampling.",
    )
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
        use_symmetry_augmentation=not args.no_symmetry_augmentation,
        seed=args.seed,
        init_checkpoint=(args.init_checkpoint or None),
        save_path=args.save_path,
    )
