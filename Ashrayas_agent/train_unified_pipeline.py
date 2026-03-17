"""
Unified Gomoku DQN Training Pipeline
=====================================

Combines two-phase training:
  Phase 1: Train agent against random opponent (GomokuEnv) for N episodes
  Phase 2: Continue training via self-play with opponent pool for M episodes

Both phases use consistent 4-plane state encoding and CNN architecture.
Output checkpoint includes metadata for reproducibility.

Usage:
  python train_unified_pipeline.py --phase1-episodes 15000 --phase2-episodes 80000
  python train_unified_pipeline.py --phase1-episodes 100 --phase2-episodes 100  # Quick test
"""

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gameboard import GomokuLogic
from gomoku_env import GomokuEnv
from gomoku_config import BOARD_SIZE


# ============================================================================
# MODELS
# ============================================================================

class SimpleDQN(nn.Module):
    """CNN-based DQN with 4-channel input (4-plane state encoding)."""
    def __init__(self, board_size=BOARD_SIZE, action_dim=None, input_channels=4):
        super().__init__()
        if action_dim is None:
            action_dim = board_size * board_size
        self.board_size = board_size
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
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


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        return torch.relu(out + x)


class ResidualDQN(nn.Module):
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


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_board, done):
        self.buffer.append((state, action, reward, next_state, next_board, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_boards, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_boards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# STATE ENCODING & ACTION SELECTION
# ============================================================================

def build_state_planes(state, current_player, last_move, board_size):
    """Build 4-plane state representation: [own, opponent, side_to_move, last_move]."""
    own = (state == current_player).astype(np.float32)
    opp = (state == -current_player).astype(np.float32)
    side_to_move_black = np.full(
        (board_size, board_size),
        1.0 if current_player == 1 else 0.0,
        dtype=np.float32,
    )
    last_move_plane = np.zeros((board_size, board_size), dtype=np.float32)
    if last_move >= 0:
        row, col = divmod(last_move, board_size)
        last_move_plane[row, col] = 1.0
    return np.stack([own, opp, side_to_move_black, last_move_plane], axis=0)


def encode_state(state, current_player, last_move, board_size):
    """Encode state as 4-plane representation (consistent for both phases)."""
    return build_state_planes(state, current_player, last_move, board_size)


def state_to_tensor(encoded_state, device):
    """Convert encoded state to PyTorch tensor."""
    return torch.as_tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)


def get_valid_actions(state):
    """Get indices of empty board positions."""
    return np.flatnonzero(state.ravel() == 0)


def select_action(policy_net, state, current_player, last_move, epsilon, device, board_size):
    """Select action using epsilon-greedy policy."""
    valid_actions = get_valid_actions(state)
    if len(valid_actions) == 0:
        return None

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    with torch.no_grad():
        encoded_state = encode_state(state, current_player, last_move, board_size)
        q_values = policy_net(state_to_tensor(encoded_state, device)).squeeze(0)
        valid_actions_t = torch.as_tensor(valid_actions, dtype=torch.long, device=device)
        best_valid_idx = torch.argmax(q_values[valid_actions_t]).item()
    return int(valid_actions[best_valid_idx])


def select_random_action(state):
    """Select a random valid action."""
    valid_actions = get_valid_actions(state)
    if len(valid_actions) == 0:
        return None
    return int(random.choice(valid_actions))


# ============================================================================
# SYMMETRY AUGMENTATION
# ============================================================================

def transform_action_index(action_idx, sym, board_size):
    """Transform action index using symmetry."""
    row, col = divmod(int(action_idx), board_size)
    if sym == 0:
        nr, nc = row, col
    elif sym == 1:
        nr, nc = board_size - 1 - col, row
    elif sym == 2:
        nr, nc = board_size - 1 - row, board_size - 1 - col
    elif sym == 3:
        nr, nc = col, board_size - 1 - row
    elif sym == 4:
        nr, nc = row, board_size - 1 - col
    elif sym == 5:
        nr, nc = board_size - 1 - row, col
    elif sym == 6:
        nr, nc = col, row
    elif sym == 7:
        nr, nc = board_size - 1 - col, board_size - 1 - row
    else:
        raise ValueError(f"Invalid symmetry id: {sym}")
    return nr * board_size + nc


def apply_symmetry_to_board(board, sym):
    """Apply symmetry transformation to board."""
    if sym == 0:
        return board
    if sym == 1:
        return np.rot90(board, 1).copy()
    if sym == 2:
        return np.rot90(board, 2).copy()
    if sym == 3:
        return np.rot90(board, 3).copy()
    if sym == 4:
        return np.fliplr(board).copy()
    if sym == 5:
        return np.flipud(board).copy()
    if sym == 6:
        return board.T.copy()
    if sym == 7:
        return np.flipud(np.fliplr(board.T)).copy()
    raise ValueError(f"Invalid symmetry id: {sym}")


def apply_symmetry_to_encoded_state(encoded_state, sym):
    """Apply symmetry to multi-channel encoded state."""
    if encoded_state.ndim == 2:
        return apply_symmetry_to_board(encoded_state, sym)
    return np.stack([apply_symmetry_to_board(channel, sym) for channel in encoded_state], axis=0)


def augment_batch_symmetry(states, actions, next_states, next_boards, board_size, channels):
    """Apply random symmetry augmentation to batch."""
    batch_size = states.shape[0]
    for i in range(batch_size):
        sym = random.randrange(8)
        if sym == 0:
            continue
        state_encoded = states[i].reshape(channels, board_size, board_size)
        next_state_encoded = next_states[i].reshape(channels, board_size, board_size)
        next_board = next_boards[i].reshape(board_size, board_size)
        states[i] = apply_symmetry_to_encoded_state(state_encoded, sym).reshape(-1)
        next_states[i] = apply_symmetry_to_encoded_state(next_state_encoded, sym).reshape(-1)
        next_boards[i] = apply_symmetry_to_board(next_board, sym).reshape(-1)
        actions[i] = transform_action_index(actions[i], sym, board_size)


# ============================================================================
# Q-LEARNING UTILITIES
# ============================================================================

def compute_next_q_max(policy_net, target_net, next_states, next_boards, dones):
    """Compute max Q-values for next states (Double DQN)."""
    with torch.no_grad():
        valid_actions_mask = next_boards.view(next_boards.size(0), -1).eq(0.0)
        next_q_policy = policy_net(next_states)
        masked_next_q_policy = next_q_policy.masked_fill(~valid_actions_mask, float("-inf"))
        next_actions = masked_next_q_policy.argmax(dim=1, keepdim=True)

        next_q_target = target_net(next_states)
        next_q_max = next_q_target.gather(1, next_actions).squeeze(1)

        no_valid_actions = ~valid_actions_mask.any(dim=1)
        next_q_max = torch.where(no_valid_actions, torch.zeros_like(next_q_max), next_q_max)
        return next_q_max * (1.0 - dones)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(path_value):
    """Resolve relative paths to PROJECT_ROOT."""
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def build_model(model_arch, board_size, action_dim):
    """Build DQN model based on architecture."""
    if model_arch == "cnn":
        return SimpleDQN(board_size=board_size, action_dim=action_dim, input_channels=4)
    if model_arch == "residual_dueling_dqn_4plane":
        return ResidualDQN(board_size=board_size, action_dim=action_dim, input_channels=4)
    raise ValueError(f"Unsupported model architecture: {model_arch}")


def copy_state_dict_to_cpu(model):
    """Copy model state dict to CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================================
# PHASE 1: TRAINING WITH RANDOM OPPONENT
# ============================================================================

def evaluate_policy_phase1(policy_net, board_size, games, device):
    """Evaluate policy in phase 1 (vs random opponent)."""
    env = GomokuEnv(size=board_size, render_mode=None)
    wins = 0
    draws = 0
    losses = 0

    for _ in range(games):
        state = env.reset().astype(np.float32)
        done = False
        info = {}
        last_move = -1

        while not done:
            action = select_action(policy_net, state, 1, last_move, epsilon=0.0, device=device, board_size=board_size)
            next_state, _, done, info = env.step(action)
            state = next_state.astype(np.float32)
            if action is not None:
                last_move = action

        result = info.get("result", "")
        if result == "Win":
            wins += 1
        elif result == "Draw":
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def train_phase1_random_opponent(
    episodes=15_000,
    board_size=BOARD_SIZE,
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    replay_capacity=100_000,
    warmup_steps=5_000,
    target_update_every=1_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=180_000,
    train_every=4,
    use_symmetry_augmentation=True,
    eval_every_episodes=500,
    eval_games=100,
    seed=42,
):
    """Phase 1: Train agent against random opponent using GomokuEnv."""
    print("\n" + "="*80)
    print("PHASE 1: Training with Random Opponent")
    print("="*80)
    
    set_global_seed(seed)
    env = GomokuEnv(size=board_size, render_mode=None)
    action_dim = env.action_space_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = build_model("cnn", board_size, action_dim).to(device)
    target_net = build_model("cnn", board_size, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    min_replay_size = max(batch_size, warmup_steps)
    best_eval_win_rate = -1.0

    global_step = 0

    for episode in range(1, episodes + 1):
        state = env.reset().astype(np.float32)
        done = False
        episode_reward = 0.0
        last_move = -1

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
                0.0, (epsilon_decay_steps - global_step) / epsilon_decay_steps
            )

            action = select_action(policy_net, state, 1, last_move, epsilon, device, board_size)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            
            if action is not None:
                last_move = action

            # Encode states with 4-plane representation
            encoded_state = encode_state(state, 1, last_move, board_size)
            encoded_next_state = encode_state(next_state, 1, last_move, board_size)
            
            replay_buffer.add(
                encoded_state.flatten(),
                action if action is not None else 0,
                reward,
                encoded_next_state.flatten(),
                next_state.flatten(),
                done
            )

            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay_buffer) >= min_replay_size and global_step % train_every == 0:
                states, actions, rewards, next_states, next_boards, dones = replay_buffer.sample(batch_size)
                channels = 4

                if use_symmetry_augmentation:
                    augment_batch_symmetry(states, actions, next_states, next_boards, board_size, channels)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                next_boards_t = torch.tensor(next_boards, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                states_t = states_t.view(batch_size, channels, board_size, board_size)
                next_states_t = next_states_t.view(batch_size, channels, board_size, board_size)

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                next_q_max = compute_next_q_max(policy_net, target_net, next_states_t, next_boards_t, dones_t)
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
                f"Phase 1 Episode {episode:>6}/{episodes} | "
                f"Reward: {episode_reward:>7.2f} | "
                f"Epsilon: {epsilon:>5.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

        if episode % eval_every_episodes == 0:
            wins, draws, losses = evaluate_policy_phase1(policy_net, board_size, eval_games, device)
            win_rate = wins / max(1, eval_games)
            print(
                f"[Phase 1 Eval @ ep {episode}] W/D/L: {wins}/{draws}/{losses} | Win rate: {win_rate:.3f}"
            )

            if win_rate > best_eval_win_rate:
                best_eval_win_rate = win_rate

    print(f"Phase 1 complete. Best win rate: {best_eval_win_rate:.3f}\n")
    return policy_net, best_eval_win_rate


# ============================================================================
# PHASE 2: SELF-PLAY TRAINING
# ============================================================================

def _max_chain_length(board, row, col, player):
    """Count max chain length of player at position."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    best = 1

    for dr, dc in directions:
        count = 1
        for i in range(1, 5):
            r, c = row + dr * i, col + dc * i
            if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                count += 1
            else:
                break
        for i in range(1, 5):
            r, c = row - dr * i, col - dc * i
            if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                count += 1
            else:
                break
        best = max(best, count)

    return best


def _board_max_chain_length(board, player):
    """Get max chain length for player on board."""
    best = 0
    positions = np.argwhere(board == player)
    for row, col in positions:
        best = max(best, _max_chain_length(board, int(row), int(col), player))
    return best


def get_immediate_winning_actions(board, player):
    """Get actions that result in immediate win."""
    valid_actions = get_valid_actions(board)
    winning_actions = []

    for action in valid_actions:
        row, col = divmod(int(action), board.shape[0])
        if board[row, col] != 0:
            continue
        board[row, col] = player
        is_win = _max_chain_length(board, row, col, player) >= 5
        board[row, col] = 0
        if is_win:
            winning_actions.append(int(action))

    return winning_actions


def select_tactical_action(policy_net, state, current_player, last_move, epsilon, device, board_size):
    """Select tactical action (win-now or block-now)."""
    own_wins = get_immediate_winning_actions(state, current_player)
    if own_wins:
        return int(random.choice(own_wins))

    opponent_wins = get_immediate_winning_actions(state, -current_player)
    if opponent_wins:
        return int(random.choice(opponent_wins))

    if policy_net is None:
        return select_random_action(state)

    return select_action(policy_net, state, current_player, last_move, epsilon, device, board_size)


def play_policy_turn(policy_net, logic, player, last_move, epsilon, device, board_size, turn_mode="policy"):
    """Execute one turn of policy play."""
    if turn_mode == "random":
        action = select_random_action(logic.board)
    elif turn_mode == "tactical":
        action = select_tactical_action(policy_net, logic.board, player, last_move, epsilon, device, board_size)
    else:
        action = select_action(policy_net, logic.board, player, last_move, epsilon, device, board_size)

    if action is None:
        return None, 0.0, True, {"result": "Draw"}

    row, col = divmod(action, logic.size)
    success, msg = logic.step(row, col, player)
    if not success:
        return action, -10.0, True, {"result": "Invalid"}

    dense_reward = 0.04 * max(0, _max_chain_length(logic.board, row, col, player) - 1)
    if msg == "Win":
        return action, 10.0, True, {"result": "Win", "dense_reward": dense_reward}
    if msg == "Draw":
        return action, 0.0, True, {"result": "Draw", "dense_reward": dense_reward}
    return action, dense_reward, False, {"result": "Continue", "dense_reward": dense_reward}


def sample_opponent_mode(opponent_pool, current_prob, pool_prob, tactical_prob):
    """Sample opponent mode based on probabilities."""
    random_prob = max(0.0, 1.0 - current_prob - pool_prob - tactical_prob)
    modes = ["current", "random"]
    weights = [max(0.0, current_prob), random_prob]

    if len(opponent_pool) > 0 and pool_prob > 0.0:
        modes.append("pool")
        weights.append(pool_prob)

    if tactical_prob > 0.0:
        modes.append("tactical")
        weights.append(tactical_prob)

    total = sum(weights)
    if total <= 0.0:
        return "random"

    normalized = [w / total for w in weights]
    return random.choices(modes, weights=normalized, k=1)[0]


def run_self_play_episode(
    policy_net,
    board_size,
    epsilon,
    step_penalty,
    device,
    opponent_net,
    opponent_mode,
    opponent_epsilon,
):
    """Run one self-play episode."""
    logic = GomokuLogic(size=board_size)
    learning_player = random.choice((1, -1))
    opponent_player = -learning_player
    episode_reward = 0.0
    move_count = 0
    transitions = []
    last_move = -1
    block_three_bonus = 0.12
    block_four_bonus = 0.35

    # Opener (if learning player is -1)
    if learning_player == -1:
        if opponent_mode == "random":
            opener_mode = "random"
            opener_net = None
            opener_eps = 1.0
        elif opponent_mode == "tactical":
            opener_mode = "tactical"
            opener_net = opponent_net
            opener_eps = opponent_epsilon
        else:
            opener_mode = "policy"
            opener_net = opponent_net
            opener_eps = opponent_epsilon

        opponent_action, opponent_reward, done, info = play_policy_turn(
            opener_net,
            logic,
            opponent_player,
            last_move,
            opener_eps,
            device,
            board_size,
            turn_mode=opener_mode,
        )
        if opponent_action is not None:
            last_move = opponent_action
        if done:
            episode_reward += -opponent_reward if info.get("result") == "Win" else opponent_reward
            return transitions, episode_reward, move_count, learning_player, opponent_mode

    done = False
    while not done:
        opponent_threat_before = _board_max_chain_length(logic.board, opponent_player)
        encoded_state = encode_state(logic.board, learning_player, last_move, board_size)
        action, own_reward, done, _ = play_policy_turn(
            policy_net, logic, learning_player, last_move, epsilon, device, board_size
        )
        move_count += 1

        if action is None:
            next_state = encode_state(logic.board, learning_player, last_move, board_size)
            transitions.append((encoded_state.flatten(), 0, 0.0, next_state.flatten(), logic.board.flatten(), True))
            break

        last_move = action
        opponent_threat_after = _board_max_chain_length(logic.board, opponent_player)
        if opponent_threat_before >= 4 and opponent_threat_after < 4:
            own_reward += block_four_bonus
        elif opponent_threat_before >= 3 and opponent_threat_after < 3:
            own_reward += block_three_bonus

        if done:
            next_state = encode_state(logic.board, learning_player, last_move, board_size)
            transitions.append(
                (encoded_state.flatten(), action, own_reward, next_state.flatten(), logic.board.flatten(), True)
            )
            episode_reward += own_reward
            break

        opponent_action, _, opponent_done, opponent_info = play_policy_turn(
            None if opponent_mode == "random" else opponent_net,
            logic,
            opponent_player,
            last_move,
            1.0 if opponent_mode == "random" else opponent_epsilon,
            device,
            board_size,
            turn_mode=("tactical" if opponent_mode == "tactical" else ("random" if opponent_mode == "random" else "policy")),
        )
        if opponent_action is not None:
            last_move = opponent_action
        next_state = encode_state(logic.board, learning_player, last_move, board_size)

        if opponent_done:
            result = opponent_info.get("result")
            reward = 0.0 if result == "Draw" else -10.0
            done = True
        else:
            if opponent_action is None:
                opponent_chain_penalty = 0.0
            else:
                row, col = divmod(opponent_action, board_size)
                opponent_chain_penalty = 0.04 * max(0, _max_chain_length(logic.board, row, col, opponent_player) - 1)
            reward = step_penalty + own_reward - opponent_chain_penalty
            done = False

        transitions.append(
            (encoded_state.flatten(), action, reward, next_state.flatten(), logic.board.flatten(), done)
        )
        episode_reward += reward

    return transitions, episode_reward, move_count, learning_player, opponent_mode


def evaluate_policy_generic(policy_net, board_size, games, device):
    """Evaluate policy in self-play (vs random opponent)."""
    wins = 0
    draws = 0
    losses = 0

    for _ in range(games):
        logic = GomokuLogic(size=board_size)
        last_move = -1
        done = False

        while not done:
            action = select_action(policy_net, logic.board, 1, last_move, epsilon=0.0, device=device, board_size=board_size)
            if action is None:
                draws += 1
                break
            row, col = divmod(action, board_size)
            _, msg = logic.step(row, col, 1)
            last_move = action

            if msg == "Win":
                wins += 1
                break
            if msg == "Draw":
                draws += 1
                break

            valid_actions = get_valid_actions(logic.board)
            if len(valid_actions) == 0:
                draws += 1
                break
            opp_action = int(random.choice(valid_actions))
            opp_row, opp_col = divmod(opp_action, board_size)
            _, msg = logic.step(opp_row, opp_col, -1)
            last_move = opp_action

            if msg == "Win":
                losses += 1
                break
            if msg == "Draw":
                draws += 1
                break

    return wins, draws, losses


def train_phase2_self_play(
    policy_net,
    episodes=80_000,
    board_size=BOARD_SIZE,
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    replay_capacity=100_000,
    warmup_steps=5_000,
    target_update_every=1_000,
    epsilon_start=0.2,
    epsilon_end=0.03,
    epsilon_decay_steps=120_000,
    train_every=4,
    step_penalty=-0.005,
    use_symmetry_augmentation=True,
    opponent_current_prob=0.4,
    opponent_pool_prob=0.4,
    opponent_tactical_prob=0.15,
    opponent_epsilon=0.03,
    opponent_pool_size=12,
    opponent_pool_update_every=500,
    eval_every_episodes=500,
    eval_games=100,
    seed=42,
    save_path="Ashrayas_agent/dqn_gomoku_unified.pt",
    best_save_path="Ashrayas_agent/dqn_gomoku_unified_best.pt",
):
    """Phase 2: Train agent via self-play against opponent pool."""
    print("\n" + "="*80)
    print("PHASE 2: Self-Play Training")
    print("="*80 + "\n")
    
    set_global_seed(seed)

    action_dim = board_size * board_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # policy_net already trained from phase 1
    target_net = build_model("cnn", board_size, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    min_replay_size = max(batch_size, warmup_steps)
    best_eval_win_rate = -1.0
    opponent_pool = deque(maxlen=opponent_pool_size)
    opponent_mode_counts = {"current": 0, "pool": 0, "random": 0, "tactical": 0}

    opponent_net = build_model("cnn", board_size, action_dim).to(device)
    opponent_net.eval()

    # Seed pool with phase 1 trained model
    opponent_pool.append(copy_state_dict_to_cpu(policy_net))

    global_step = 0

    resolved_save_path = resolve_project_path(save_path)
    resolved_best_save_path = resolve_project_path(best_save_path)
    resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_best_save_path.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, episodes + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, (epsilon_decay_steps - global_step) / epsilon_decay_steps
        )

        if opponent_pool_update_every > 0 and episode % opponent_pool_update_every == 0:
            opponent_pool.append(copy_state_dict_to_cpu(policy_net))

        opponent_mode = sample_opponent_mode(
            opponent_pool,
            opponent_current_prob,
            opponent_pool_prob,
            opponent_tactical_prob,
        )
        if opponent_mode == "pool" and len(opponent_pool) > 0:
            sampled_state_dict = random.choice(list(opponent_pool))
            opponent_net.load_state_dict(sampled_state_dict)
            opponent_net.eval()
            selected_opponent_net = opponent_net
        elif opponent_mode == "current":
            selected_opponent_net = policy_net
        elif opponent_mode == "tactical":
            selected_opponent_net = policy_net
        else:
            selected_opponent_net = None

        transitions, episode_reward, move_count, learning_player, used_opponent_mode = run_self_play_episode(
            policy_net,
            board_size,
            epsilon,
            step_penalty,
            device,
            selected_opponent_net,
            opponent_mode,
            opponent_epsilon,
        )
        opponent_mode_counts[used_opponent_mode] += 1

        for transition in transitions:
            replay_buffer.add(*transition)
            global_step += 1

            if len(replay_buffer) >= min_replay_size and global_step % train_every == 0:
                states, actions, rewards, next_states, next_boards, dones = replay_buffer.sample(batch_size)
                channels = 4

                if use_symmetry_augmentation:
                    augment_batch_symmetry(states, actions, next_states, next_boards, board_size, channels)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                next_boards_t = torch.tensor(next_boards, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                states_t = states_t.view(batch_size, channels, board_size, board_size)
                next_states_t = next_states_t.view(batch_size, channels, board_size, board_size)

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                next_q_max = compute_next_q_max(policy_net, target_net, next_states_t, next_boards_t, dones_t)
                q_target = rewards_t + gamma * next_q_max

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0 or episode == 1:
            learner_side = "Black" if learning_player == 1 else "White"
            print(
                f"Phase 2 Episode {episode:>6}/{episodes} | "
                f"Learner: {learner_side:<5} | "
                f"Opp: {used_opponent_mode:<7} | "
                f"Moves: {move_count:>3} | "
                f"Reward: {episode_reward:>7.2f} | "
                f"Epsilon: {epsilon:>5.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

        if episode % eval_every_episodes == 0:
            wins, draws, losses = evaluate_policy_generic(
                policy_net, board_size, eval_games, device
            )
            win_rate = wins / max(1, eval_games)
            print(
                f"[Phase 2 Eval @ ep {episode}] W/D/L: {wins}/{draws}/{losses} "
                f"| Win rate: {win_rate:.3f} "
                f"| Opp mix (cur/pool/rnd/tac): "
                f"{opponent_mode_counts['current']}/"
                f"{opponent_mode_counts['pool']}/"
                f"{opponent_mode_counts['random']}"
                f"/{opponent_mode_counts['tactical']}"
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
                        "training_mode": "unified_pipeline_phase2",
                        "opponent_current_prob": opponent_current_prob,
                        "opponent_pool_prob": opponent_pool_prob,
                        "opponent_tactical_prob": opponent_tactical_prob,
                        "opponent_epsilon": opponent_epsilon,
                        "opponent_pool_size": opponent_pool_size,
                    },
                    resolved_best_save_path,
                )
                print(f"New best model saved to: {resolved_best_save_path}")

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "model_arch": "cnn",
            "board_size": board_size,
            "action_dim": action_dim,
            "best_eval_win_rate": best_eval_win_rate,
            "seed": seed,
            "training_mode": "unified_pipeline_phase2",
            "opponent_current_prob": opponent_current_prob,
            "opponent_pool_prob": opponent_pool_prob,
            "opponent_tactical_prob": opponent_tactical_prob,
            "opponent_epsilon": opponent_epsilon,
            "opponent_pool_size": opponent_pool_size,
        },
        resolved_save_path,
    )
    print(f"\nPhase 2 complete. Model saved to: {resolved_save_path}")
    if best_eval_win_rate >= 0.0:
        print(f"Best eval win rate: {best_eval_win_rate:.3f} | Best model path: {resolved_best_save_path}\n")

    return policy_net, best_eval_win_rate


# ============================================================================
# UNIFIED PIPELINE
# ============================================================================

def train_unified_pipeline(
    phase1_episodes=15_000,
    phase2_episodes=80_000,
    board_size=BOARD_SIZE,
    gamma=0.99,
    lr=1e-4,
    batch_size=128,
    replay_capacity=100_000,
    warmup_steps=5_000,
    target_update_every=1_000,
    phase1_epsilon_start=1.0,
    phase1_epsilon_end=0.05,
    phase1_epsilon_decay_steps=180_000,
    phase2_epsilon_start=0.2,
    phase2_epsilon_end=0.03,
    phase2_epsilon_decay_steps=120_000,
    train_every=4,
    step_penalty=-0.005,
    use_symmetry_augmentation=True,
    opponent_current_prob=0.4,
    opponent_pool_prob=0.4,
    opponent_tactical_prob=0.15,
    opponent_epsilon=0.03,
    opponent_pool_size=12,
    opponent_pool_update_every=500,
    eval_every_episodes=500,
    eval_games=100,
    seed=42,
    save_path="Ashrayas_agent/dqn_gomoku_unified.pt",
    best_save_path="Ashrayas_agent/dqn_gomoku_unified_best.pt",
):
    """Unified pipeline: Phase 1 (random opponent) -> Phase 2 (self-play)."""
    print("\n\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "UNIFIED GOMOKU DQN TRAINING PIPELINE".center(78) + "║")
    print("║" + f"Phase 1: {phase1_episodes:,} episodes (Random Opponent)".center(78) + "║")
    print("║" + f"Phase 2: {phase2_episodes:,} episodes (Self-Play)".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝\n")

    # Phase 1: Train with random opponent
    policy_net, phase1_win_rate = train_phase1_random_opponent(
        episodes=phase1_episodes,
        board_size=board_size,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        warmup_steps=warmup_steps,
        target_update_every=target_update_every,
        epsilon_start=phase1_epsilon_start,
        epsilon_end=phase1_epsilon_end,
        epsilon_decay_steps=phase1_epsilon_decay_steps,
        train_every=train_every,
        use_symmetry_augmentation=use_symmetry_augmentation,
        eval_every_episodes=eval_every_episodes,
        eval_games=eval_games,
        seed=seed,
    )

    # Phase 2: Self-play training
    policy_net, phase2_win_rate = train_phase2_self_play(
        policy_net=policy_net,
        episodes=phase2_episodes,
        board_size=board_size,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        warmup_steps=warmup_steps,
        target_update_every=target_update_every,
        epsilon_start=phase2_epsilon_start,
        epsilon_end=phase2_epsilon_end,
        epsilon_decay_steps=phase2_epsilon_decay_steps,
        train_every=train_every,
        step_penalty=step_penalty,
        use_symmetry_augmentation=use_symmetry_augmentation,
        opponent_current_prob=opponent_current_prob,
        opponent_pool_prob=opponent_pool_prob,
        opponent_tactical_prob=opponent_tactical_prob,
        opponent_epsilon=opponent_epsilon,
        opponent_pool_size=opponent_pool_size,
        opponent_pool_update_every=opponent_pool_update_every,
        eval_every_episodes=eval_every_episodes,
        eval_games=eval_games,
        seed=seed,
        save_path=save_path,
        best_save_path=best_save_path,
    )

    # Print summary
    print("\n" + "="*80)
    print("UNIFIED TRAINING COMPLETE")
    print("="*80)
    print(f"Phase 1 best win rate: {phase1_win_rate:.3f}")
    print(f"Phase 2 best win rate: {phase2_win_rate:.3f}")
    print(f"Final model saved to: {resolve_project_path(best_save_path)}")
    print("="*80 + "\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Gomoku DQN Training Pipeline: Phase 1 (Random Opponent) + Phase 2 (Self-Play)"
    )
    # Phase configuration
    parser.add_argument("--phase1-episodes", type=int, default=15_000,
                        help="Number of episodes for phase 1 (random opponent training)")
    parser.add_argument("--phase2-episodes", type=int, default=80_000,
                        help="Number of episodes for phase 2 (self-play training)")
    
    # Shared training params
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--target-update-every", type=int, default=1_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Phase 1 epsilon schedule
    parser.add_argument("--phase1-epsilon-start", type=float, default=1.0)
    parser.add_argument("--phase1-epsilon-end", type=float, default=0.05)
    parser.add_argument("--phase1-epsilon-decay-steps", type=int, default=180_000)
    
    # Phase 2 epsilon schedule
    parser.add_argument("--phase2-epsilon-start", type=float, default=0.2)
    parser.add_argument("--phase2-epsilon-end", type=float, default=0.03)
    parser.add_argument("--phase2-epsilon-decay-steps", type=int, default=120_000)
    
    # Phase 2 opponent configuration
    parser.add_argument("--step-penalty", type=float, default=-0.005)
    parser.add_argument("--opponent-current-prob", type=float, default=0.4,
                        help="Probability of using current policy as opponent")
    parser.add_argument("--opponent-pool-prob", type=float, default=0.4,
                        help="Probability of using pool opponent")
    parser.add_argument("--opponent-tactical-prob", type=float, default=0.15,
                        help="Probability of using tactical opponent")
    parser.add_argument("--opponent-epsilon", type=float, default=0.03)
    parser.add_argument("--opponent-pool-size", type=int, default=12)
    parser.add_argument("--opponent-pool-update-every", type=int, default=500)
    
    # Evaluation
    parser.add_argument("--eval-every-episodes", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=100)
    
    # Augmentation
    parser.add_argument("--no-symmetry-augmentation", action="store_true",
                        help="Disable symmetry augmentation")
    
    # Output paths
    parser.add_argument("--save-path", type=str, default="Ashrayas_agent/dqn_gomoku_unified.pt")
    parser.add_argument("--best-save-path", type=str, default="Ashrayas_agent/dqn_gomoku_unified_best.pt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_unified_pipeline(
        phase1_episodes=args.phase1_episodes,
        phase2_episodes=args.phase2_episodes,
        board_size=args.board_size,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        target_update_every=args.target_update_every,
        phase1_epsilon_start=args.phase1_epsilon_start,
        phase1_epsilon_end=args.phase1_epsilon_end,
        phase1_epsilon_decay_steps=args.phase1_epsilon_decay_steps,
        phase2_epsilon_start=args.phase2_epsilon_start,
        phase2_epsilon_end=args.phase2_epsilon_end,
        phase2_epsilon_decay_steps=args.phase2_epsilon_decay_steps,
        train_every=args.train_every,
        step_penalty=args.step_penalty,
        use_symmetry_augmentation=not args.no_symmetry_augmentation,
        opponent_current_prob=args.opponent_current_prob,
        opponent_pool_prob=args.opponent_pool_prob,
        opponent_tactical_prob=args.opponent_tactical_prob,
        opponent_epsilon=args.opponent_epsilon,
        opponent_pool_size=args.opponent_pool_size,
        opponent_pool_update_every=args.opponent_pool_update_every,
        eval_every_episodes=args.eval_every_episodes,
        eval_games=args.eval_games,
        seed=args.seed,
        save_path=args.save_path,
        best_save_path=args.best_save_path,
    )
