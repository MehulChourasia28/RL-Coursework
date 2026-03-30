"""
AlphaZero training for 9x9 Gomoku.

Usage (from repo root):
    python -m Mehuls_agent.train
    python -m Mehuls_agent.train --no-resume

Algorithm:
    1. Self-play with MCTS to generate (state, policy, value) triples
    2. Augment each game with 8 board symmetries (4 rotations × 2 flips)
    3. Train the PolicyValueNet on the replay buffer
    4. Repeat

Hyperparameter guide:
    N_SIMS_TRAIN = 200  — quality/speed tradeoff; try 100 for faster training
    GAMES_PER_ITER     — more games → more diverse data; try 50 if GPU is slow
    N_ITERS            — 50 iters is typically enough for a strong 9x9 bot
"""

import argparse
import os
import random
import sys
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

from gameboard import GomokuLogic
from Mehuls_agent.model import PolicyValueNet, encode_state
from Mehuls_agent.mcts import MCTS

BOARD_SIZE = 9
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Benchmarked on this machine (64ch/4-block model, torch.compile):
#   inference @ batch=1:             ~0.5 ms   (vs ~1.6 ms for 128ch/6-block)
#   1 game @ 400 sims + tree reuse:  ~3-4 s
#   50 games/iter self-play:         ~3 min/iter
#   80 iters total:                  ~4 h
#
# Key changes vs previous run:
#   - 64ch/4-block model: ~3x faster inference → 400 sims in same time as 100
#   - Tree reuse: subtree is kept after each move, so later moves in a game
#     accumulate sims from all prior searches — much stronger policy targets
#   - TEMP_THRESHOLD 10→15: more diverse openings before greedy play
#   - Resign: saves ~30% of self-play time, generates more varied games
N_ITERS        = 500           # training iterations
GAMES_PER_ITER = 50           # self-play games per iteration
N_SIMS_TRAIN   = 200          # MCTS simulations per move (self-play)
N_SIMS_EVAL    = 400          # MCTS simulations per move (evaluation)
TEMP_THRESHOLD = 8            # moves before temperature drops to 0 (~15% of avg game)

RESIGN_THRESHOLD = 0.95       # resign if root_value < -RESIGN_THRESHOLD
RESIGN_PATIENCE  = 5          # consecutive moves below threshold before resign
RESIGN_MIN_MOVE  = 20         # don't resign before this move number
RESIGN_RATE      = 0.0        # disabled — value head must mature before resign is safe

HEURISTIC_GAME_RATE = 0.3     # fraction of games played vs heuristic opponent

TRAIN_STEPS    = 300          # gradient steps per iteration
BATCH_SIZE     = 512
LR             = 2e-3
LR_MIN         = 1e-4
L2_REG         = 1e-4
VALUE_LOSS_WEIGHT = 2.0       # upweight value head (policy gradients otherwise dominate)
BUFFER_SIZE    = 200_000

EVAL_FREQ      = 5            # evaluate every N iterations
SAVE_FREQ      = 5
EVAL_GAMES     = 40           # games per evaluation

# Network architecture
CHANNELS       = 64
RES_BLOCKS     = 4


# =========================================================================
# Board symmetry augmentation (D4 group: 4 rotations × 2 reflections)
# =========================================================================

def augment(
    states: List[np.ndarray],
    policies: List[np.ndarray],
    values: List[float],
) -> Tuple[List, List, List]:
    """Return 8× data by applying all dihedral board symmetries."""
    sz = BOARD_SIZE
    aug_s, aug_p, aug_v = [], [], []
    for s, p, v in zip(states, policies, values):
        p_grid = p.reshape(sz, sz)
        for k in range(4):
            # rotate k×90°
            s_r = np.rot90(s, k, axes=(1, 2)).copy()
            p_r = np.rot90(p_grid, k).flatten().copy()
            aug_s.append(s_r); aug_p.append(p_r); aug_v.append(v)
            # flip left-right after rotation
            s_f = np.flip(s_r, axis=2).copy()
            p_f = np.flip(p_r.reshape(sz, sz), axis=1).flatten().copy()
            aug_s.append(s_f); aug_p.append(p_f); aug_v.append(v)
    return aug_s, aug_p, aug_v


# =========================================================================
# Self-play game generation
# =========================================================================

def self_play_game(mcts: MCTS, allow_resign: bool = True) -> List[Tuple]:
    """
    Play one full game using MCTS and return training examples.

    Tree reuse: mcts.advance(action) is called after every move so that
    subsequent searches build on prior simulations rather than starting fresh.

    Resign: if root_value < -RESIGN_THRESHOLD for RESIGN_PATIENCE consecutive
    moves (after RESIGN_MIN_MOVE), the current player concedes.  The final
    outcome is assigned accordingly.  10% of games disable resign to prevent
    the network overfitting to "resignable" positions.

    Returns:
        list of (encoded_state, mcts_policy, outcome)
        outcome: +1 = current player won, -1 = lost, 0 = draw
    """
    logic   = GomokuLogic(BOARD_SIZE)
    player  = 1        # Black always moves first
    history = []       # (encoded_state, mcts_policy, player_at_step)
    move_n  = 0
    consec_losing = 0
    winner  = 0        # default draw

    mcts.reset()       # discard any tree from a previous game

    while not logic.game_over:
        board = logic.board.copy()
        temp  = 1.0 if move_n < TEMP_THRESHOLD else 0.0

        policy = mcts.get_policy(board, player, temperature=temp, add_noise=True)

        # Mask and renormalise over legal moves
        valid = (board.flatten() == 0).astype(np.float32)
        p_masked = policy * valid
        s = p_masked.sum()
        if s < 1e-10:
            break
        p_masked /= s

        history.append((encode_state(board, player), policy.copy(), player))

        # Resign check (uses MCTS-backed value at root after search)
        if allow_resign and move_n >= RESIGN_MIN_MOVE:
            if mcts.root_value < -RESIGN_THRESHOLD:
                consec_losing += 1
            else:
                consec_losing = 0
            if consec_losing >= RESIGN_PATIENCE:
                winner = -player   # current player resigns, opponent wins
                break

        action = int(np.random.choice(len(p_masked), p=p_masked))
        row, col = divmod(action, BOARD_SIZE)
        logic.step(row, col, player)

        # Reuse subtree: advance root to the played child
        mcts.advance(action)

        player  = -player
        move_n += 1

    if winner == 0:
        winner = logic.winner  # +1, -1, or 0 (draw / board full)

    data = []
    for state, pol, p in history:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == p else -1.0
        data.append((state, pol, value))
    return data


# =========================================================================
# Heuristic game generation
# =========================================================================

def heuristic_game(mcts: MCTS, agent_color: int) -> List[Tuple]:
    """
    Play one game: MCTS agent (agent_color) vs heuristic opponent.
    Returns training examples only for the agent's moves.

    This is critical for teaching the agent to handle tactical threats
    that don't emerge in pure self-play (instant win/block, threat chains).
    """
    from Mehuls_agent.heuristics import heuristic_move

    logic   = GomokuLogic(BOARD_SIZE)
    player  = 1
    history = []
    move_n  = 0

    mcts.reset()

    while not logic.game_over:
        board = logic.board.copy()

        if player == agent_color:
            temp = 1.0 if move_n < TEMP_THRESHOLD else 0.0
            policy = mcts.get_policy(board, player, temperature=temp, add_noise=True)
            valid  = (board.flatten() == 0).astype(np.float32)
            p_masked = policy * valid
            s = p_masked.sum()
            if s < 1e-10:
                break
            p_masked /= s
            history.append((encode_state(board, player), policy.copy(), player))
            action = int(np.random.choice(len(p_masked), p=p_masked))
        else:
            r, c   = heuristic_move(board, player)
            action = r * BOARD_SIZE + c

        row, col = divmod(action, BOARD_SIZE)
        logic.step(row, col, player)
        mcts.advance(action)
        player  = -player
        move_n += 1

    winner = logic.winner
    data   = []
    for state, pol, p in history:
        value = 0.0 if winner == 0 else (1.0 if winner == p else -1.0)
        data.append((state, pol, value))
    return data


# =========================================================================
# Training step
# =========================================================================

def train_step(
    net: PolicyValueNet,
    optimizer: optim.Optimizer,
    buffer: list,
) -> Tuple[float, float]:
    if len(buffer) < BATCH_SIZE:
        return 0.0, 0.0

    idx      = random.sample(range(len(buffer)), BATCH_SIZE)
    states   = torch.FloatTensor(np.stack([buffer[i][0] for i in idx])).to(net._device)
    policies = torch.FloatTensor(np.stack([buffer[i][1] for i in idx])).to(net._device)
    values   = torch.FloatTensor(np.array([buffer[i][2] for i in idx])).to(net._device)

    net.train()
    log_p, v = net(states)

    policy_loss = -(policies * log_p).sum(dim=1).mean()
    value_loss  = ((v - values) ** 2).mean()
    loss        = policy_loss + VALUE_LOSS_WEIGHT * value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item())


# =========================================================================
# Evaluation helpers
# =========================================================================

def _eval_vs(net: PolicyValueNet, device: torch.device, opponent: str, n: int) -> float:
    from Mehuls_agent.heuristics import heuristic_move
    mcts_eval = MCTS(net, device, n_simulations=N_SIMS_EVAL, dirichlet_epsilon=0.0)
    net.eval()
    wins = 0
    for game_i in range(n):
        logic       = GomokuLogic(BOARD_SIZE)
        agent_color = 1 if game_i % 2 == 0 else -1
        cur_player  = 1
        mcts_eval.reset()   # discard tree from previous game
        while not logic.game_over:
            board = logic.board.copy()
            if cur_player == agent_color:
                policy = mcts_eval.get_policy(board, cur_player, temperature=0.0,
                                              add_noise=False)
                valid  = (board.flatten() == 0).astype(np.float32)
                policy = policy * valid
                action = int(np.argmax(policy))
            elif opponent == "random":
                empty  = list(zip(*np.where(board == 0)))
                r, c   = random.choice(empty)
                action = int(r) * BOARD_SIZE + int(c)
            else:   # heuristic
                r, c   = heuristic_move(board, cur_player)
                action = r * BOARD_SIZE + c
            r, c = divmod(action, BOARD_SIZE)
            logic.step(r, c, cur_player)
            mcts_eval.advance(action)   # keep tree in sync for both players' moves
            cur_player = -cur_player
        if logic.winner == agent_color:
            wins += 1
    return wins / n


# =========================================================================
# Checkpoint helpers
# =========================================================================

def _strip(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _save(net, optimizer, scheduler, iteration):
    torch.save({
        "net":       _strip(net.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "config":    {"channels": CHANNELS, "num_res_blocks": RES_BLOCKS,
                      "board_size": BOARD_SIZE},
    }, CHECKPOINT_PATH)


# =========================================================================
# Training loop
# =========================================================================

def train(resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AlphaZero Gomoku {BOARD_SIZE}×{BOARD_SIZE} | device={device}")

    net = PolicyValueNet(BOARD_SIZE, 3, CHANNELS, RES_BLOCKS).to(device)
    net._device = device   # stash for train_step

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=L2_REG)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_ITERS, eta_min=LR_MIN)

    start_iter = 1
    if resume and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        # Only resume if checkpoint is an AlphaZero checkpoint (has 'config' key)
        if "config" in ckpt:
            net.load_state_dict(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_iter = ckpt.get("iteration", 1) + 1
            print(f"  Resumed from iteration {start_iter - 1}")
        else:
            print("  Found old DDQN checkpoint — starting from scratch")
    else:
        print("  Starting from scratch")

    try:
        net = torch.compile(net)
        print("  torch.compile enabled")
    except Exception:
        pass

    mcts_train = MCTS(net, device, n_simulations=N_SIMS_TRAIN)
    buffer     = deque(maxlen=BUFFER_SIZE)
    t0         = time.time()

    try:
        for it in range(start_iter, N_ITERS + 1):

            # ── Self-play ─────────────────────────────────────────────────
            net.eval()
            game_lens = []
            for _ in range(GAMES_PER_ITER):
                if random.random() < HEURISTIC_GAME_RATE:
                    agent_color = random.choice([1, -1])
                    data = heuristic_game(mcts_train, agent_color)
                else:
                    allow_resign = (random.random() < RESIGN_RATE)
                    data = self_play_game(mcts_train, allow_resign=allow_resign)
                ss = [d[0] for d in data]
                ps = [d[1] for d in data]
                vs = [d[2] for d in data]
                aug_s, aug_p, aug_v = augment(ss, ps, vs)
                for s, p, v in zip(aug_s, aug_p, aug_v):
                    buffer.append((s, p, v))
                game_lens.append(len(data))

            # ── Training ─────────────────────────────────────────────────
            pl_list, vl_list = [], []
            for _ in range(TRAIN_STEPS):
                pl, vl = train_step(net, optimizer, list(buffer))
                pl_list.append(pl)
                vl_list.append(vl)
            scheduler.step()

            avg_len = float(np.mean(game_lens))
            avg_pl  = float(np.mean(pl_list)) if pl_list else 0.0
            avg_vl  = float(np.mean(vl_list)) if vl_list else 0.0
            print(
                f"Iter {it:3d}/{N_ITERS} | "
                f"avgGameLen={avg_len:.1f} | "
                f"pl={avg_pl:.4f} vl={avg_vl:.4f} | "
                f"buf={len(buffer):6d} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{time.time()-t0:.0f}s"
            )

            # ── Evaluation ───────────────────────────────────────────────
            if it % EVAL_FREQ == 0:
                wr = _eval_vs(net, device, "random",    EVAL_GAMES)
                wh = _eval_vs(net, device, "heuristic", EVAL_GAMES)
                print(f"  [eval] vs random={wr:.2f}  vs heuristic={wh:.2f}")

            # ── Checkpoint ───────────────────────────────────────────────
            if it % SAVE_FREQ == 0:
                _save(net, optimizer, scheduler, it)
                print(f"  [saved] iter {it}")

    except KeyboardInterrupt:
        print("\nInterrupted — saving…")
        _save(net, optimizer, scheduler, it)
        print(f"Done. Checkpoint saved to {CHECKPOINT_PATH}")
        return

    _save(net, optimizer, scheduler, N_ITERS)
    print(f"Done. Checkpoint saved to {CHECKPOINT_PATH}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    train(resume=not args.no_resume)
