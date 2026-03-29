"""
AlphaZero Gomoku – study training script.

Runs one named configuration driven by a JSON config file.
All outputs go to  study/runs/<name>/
  log.txt        – human-readable progress
  metrics.jsonl  – one JSON object per iteration (for analysis)
  checkpoint_NNNN.pt  – periodic checkpoints
  checkpoint_latest.pt

Usage (from repo root):
    python study/train_run.py --config study/configs/cfg1.json
"""

import argparse
import json
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
sys.stdout.reconfigure(line_buffering=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from gameboard import GomokuLogic
from Mehuls_agent.model import PolicyValueNet, encode_state

try:
    from Mehuls_agent.mcts_fast import FastMCTS as MCTS
    print("MCTS backend: C++ (FastMCTS)")
except ImportError:
    from Mehuls_agent.mcts import MCTS
    print("MCTS backend: Python (fallback)")

BOARD_SIZE = 9


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler factory
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg):
    s = cfg["lr_schedule"]
    if s["type"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["n_iters"], eta_min=s["eta_min"]
        )
    elif s["type"] == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=s["milestones"], gamma=s["gamma"]
        )
    raise ValueError(f"Unknown lr_schedule type: {s['type']}")


# ─────────────────────────────────────────────────────────────────────────────
# Progressive MCTS sim schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_n_sims(it: int, cfg: dict) -> int:
    sims = cfg["n_sims_train"]
    if isinstance(sims, list):
        n_phases = len(sims)
        phase = min(int((it - 1) * n_phases / cfg["n_iters"]), n_phases - 1)
        return sims[phase]
    return int(sims)


# ─────────────────────────────────────────────────────────────────────────────
# Data augmentation  (D4 dihedral group: 4 rotations × 2 reflections)
# ─────────────────────────────────────────────────────────────────────────────

def augment(
    states: List[np.ndarray],
    policies: List[np.ndarray],
    values: List[float],
) -> Tuple[List, List, List]:
    sz = BOARD_SIZE
    aug_s, aug_p, aug_v = [], [], []
    for s, p, v in zip(states, policies, values):
        p_grid = p.reshape(sz, sz)
        for k in range(4):
            s_r = np.rot90(s, k, axes=(1, 2)).copy()
            p_r = np.rot90(p_grid, k).flatten().copy()
            aug_s.append(s_r); aug_p.append(p_r); aug_v.append(v)
            s_f = np.flip(s_r, axis=2).copy()
            p_f = np.flip(p_r.reshape(sz, sz), axis=1).flatten().copy()
            aug_s.append(s_f); aug_p.append(p_f); aug_v.append(v)
    return aug_s, aug_p, aug_v


# ─────────────────────────────────────────────────────────────────────────────
# Self-play game generation
# ─────────────────────────────────────────────────────────────────────────────

def self_play_game(mcts: MCTS, cfg: dict) -> List[Tuple]:
    logic = GomokuLogic(BOARD_SIZE)
    player, move_n, history = 1, 0, []
    mcts.reset()

    while not logic.game_over:
        board = logic.board.copy()
        temp  = 1.0 if move_n < cfg["temp_threshold"] else 0.0
        policy = mcts.get_policy(board, player, temperature=temp, add_noise=True)

        valid = (board.flatten() == 0).astype(np.float32)
        p_m   = policy * valid
        total = p_m.sum()
        if total < 1e-10:
            break
        p_m /= total

        history.append((encode_state(board, player), policy.copy(), player))
        action = int(np.random.choice(len(p_m), p=p_m))
        r, c   = divmod(action, BOARD_SIZE)
        logic.step(r, c, player)
        mcts.advance(action)
        player  = -player
        move_n += 1

    winner = logic.winner
    return [
        (st, po, 0.0 if winner == 0 else (1.0 if winner == pl else -1.0))
        for st, po, pl in history
    ]


def heuristic_game(mcts: MCTS, agent_color: int, cfg: dict) -> List[Tuple]:
    from Mehuls_agent.heuristics import heuristic_move
    logic = GomokuLogic(BOARD_SIZE)
    player, move_n, history = 1, 0, []
    mcts.reset()

    while not logic.game_over:
        board = logic.board.copy()
        if player == agent_color:
            temp   = 1.0 if move_n < cfg["temp_threshold"] else 0.0
            policy = mcts.get_policy(board, player, temperature=temp, add_noise=True)
            valid  = (board.flatten() == 0).astype(np.float32)
            p_m    = policy * valid
            total  = p_m.sum()
            if total < 1e-10:
                break
            p_m /= total
            history.append((encode_state(board, player), policy.copy(), player))
            action = int(np.random.choice(len(p_m), p=p_m))
        else:
            r, c   = heuristic_move(board, player)
            action = r * BOARD_SIZE + c

        r, c = divmod(action, BOARD_SIZE)
        logic.step(r, c, player)
        mcts.advance(action)
        player  = -player
        move_n += 1

    winner = logic.winner
    return [
        (st, po, 0.0 if winner == 0 else (1.0 if winner == pl else -1.0))
        for st, po, pl in history
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_step(
    net: PolicyValueNet,
    optimizer: optim.Optimizer,
    buffer: list,
    cfg: dict,
    device: torch.device,
) -> Tuple[float, float]:
    bs = cfg["batch_size"]
    if len(buffer) < bs:
        return 0.0, 0.0

    idx      = random.sample(range(len(buffer)), bs)
    states   = torch.FloatTensor(np.stack([buffer[i][0] for i in idx])).to(device)
    policies = torch.FloatTensor(np.stack([buffer[i][1] for i in idx])).to(device)
    values   = torch.FloatTensor(np.array( [buffer[i][2] for i in idx])).to(device)

    net.train()
    log_p, v    = net(states)
    policy_loss = -(policies * log_p).sum(dim=1).mean()
    value_loss  = ((v - values) ** 2).mean()
    loss        = policy_loss + cfg["value_loss_weight"] * value_loss

    optimizer.zero_grad()
    loss.backward()
    clip = cfg.get("grad_clip")
    if clip:
        nn.utils.clip_grad_norm_(net.parameters(), float(clip))
    optimizer.step()

    return float(policy_loss), float(value_loss)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_vs(net: PolicyValueNet, device: torch.device,
            opponent: str, cfg: dict) -> float:
    from Mehuls_agent.heuristics import heuristic_move
    n_sims = cfg["n_sims_eval"]
    mcts_e = MCTS(net, device, n_simulations=n_sims, dirichlet_epsilon=0.0)
    net.eval()
    wins = 0
    for gi in range(cfg["eval_games"]):
        logic       = GomokuLogic(BOARD_SIZE)
        agent_color = 1 if gi % 2 == 0 else -1
        cur         = 1
        mcts_e.reset()
        while not logic.game_over:
            board = logic.board.copy()
            if cur == agent_color:
                policy = mcts_e.get_policy(board, cur, temperature=0.0, add_noise=False)
                valid  = (board.flatten() == 0).astype(np.float32)
                action = int(np.argmax(policy * valid))
            elif opponent == "random":
                empty  = list(zip(*np.where(board == 0)))
                r, c   = random.choice(empty)
                action = int(r) * BOARD_SIZE + int(c)
            else:
                r, c   = heuristic_move(board, cur)
                action = r * BOARD_SIZE + c
            r, c = divmod(action, BOARD_SIZE)
            logic.step(r, c, cur)
            mcts_e.advance(action)
            cur = -cur
        wins += int(logic.winner == agent_color)
    return wins / cfg["eval_games"]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helper
# ─────────────────────────────────────────────────────────────────────────────

def _strip(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def save_checkpoint(net, optimizer, scheduler, it, cfg, out_dir, tag=None):
    payload = {
        "net":       _strip(net.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": it,
        "config":    cfg,
    }
    if tag is not None:
        torch.save(payload, os.path.join(out_dir, f"checkpoint_{tag}.pt"))
    torch.save(payload, os.path.join(out_dir, "checkpoint_latest.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    name    = cfg["name"]
    out_dir = os.path.join(REPO, "study", "runs", name)
    os.makedirs(out_dir, exist_ok=True)

    log_f     = open(os.path.join(out_dir, "log.txt"),      "w", buffering=1)
    metrics_f = open(os.path.join(out_dir, "metrics.jsonl"), "w", buffering=1)

    sep = "─" * 70

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")

    log(sep)
    log(f"  AlphaZero Gomoku – Study Run: {name}")
    log(sep)
    log(json.dumps(cfg, indent=2))
    log(sep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")

    net = PolicyValueNet(
        board_size      = BOARD_SIZE,
        in_channels     = 3,
        channels        = cfg["channels"],
        num_res_blocks  = cfg["res_blocks"],
        value_fc_hidden = cfg.get("value_fc_hidden", 64),
    ).to(device)
    net._device = device

    n_params = sum(p.numel() for p in net.parameters())
    log(f"Params : {n_params:,}  ({cfg['res_blocks']} res-blocks, {cfg['channels']} ch, "
        f"value-fc {cfg.get('value_fc_hidden', 64)})")

    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("l2_reg", 1e-4),
    )
    scheduler = build_scheduler(optimizer, cfg)

    try:
        net = torch.compile(net)
        log("torch.compile : enabled")
    except Exception:
        log("torch.compile : unavailable")

    buffer  = deque(maxlen=cfg["buffer_size"])
    n_iters  = cfg["n_iters"]        # used for LR scheduler T_max / milestones
    stop_at  = cfg.get("stop_iter", n_iters)  # actual training cutoff
    c_puct   = cfg.get("c_puct", 1.5)

    cur_sims = get_n_sims(1, cfg)

    # Single MCTS instance shared across all games in a run, reset between games.
    # Identical to the original train.py approach.
    mcts = MCTS(net, device, n_simulations=cur_sims, c_puct=c_puct)

    log(sep)
    log(f"  Training for {stop_at}/{n_iters} iterations  |  "
        f"games/iter = {cfg['games_per_iter']}  |  "
        f"initial sims/move = {cur_sims}  |  sequential")
    log(sep)

    t0 = time.time()

    for it in range(1, stop_at + 1):

        # ── Update sim count if schedule changed ──────────────────────────
        new_sims = get_n_sims(it, cfg)
        if new_sims != cur_sims:
            log(f"  [sched] MCTS sims: {cur_sims} → {new_sims}")
            cur_sims = new_sims
            mcts = MCTS(net, device, n_simulations=cur_sims, c_puct=c_puct)

        # ── Sequential self-play (identical to original train.py) ─────────
        net.eval()
        game_lens = []
        heur_rate = cfg.get("heuristic_game_rate", 0.3)
        n_games   = cfg["games_per_iter"]

        for _ in range(n_games):
            if random.random() < heur_rate:
                data = heuristic_game(mcts, random.choice([1, -1]), cfg)
            else:
                data = self_play_game(mcts, cfg)

            ss = [d[0] for d in data]
            ps = [d[1] for d in data]
            vs = [d[2] for d in data]
            aug_s, aug_p, aug_v = augment(ss, ps, vs)
            for s, p, v in zip(aug_s, aug_p, aug_v):
                buffer.append((s, p, v))
            game_lens.append(len(data))

        # ── Gradient updates ──────────────────────────────────────────────
        pl_acc, vl_acc = [], []
        buf_list = list(buffer)
        for _ in range(cfg["train_steps"]):
            pl, vl = train_step(net, optimizer, buf_list, cfg, device)
            pl_acc.append(pl)
            vl_acc.append(vl)
        scheduler.step()

        avg_len = float(np.mean(game_lens))
        avg_pl  = float(np.mean(pl_acc)) if pl_acc else 0.0
        avg_vl  = float(np.mean(vl_acc)) if vl_acc else 0.0
        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        log(
            f"Iter {it:3d}/{n_iters} | "
            f"gameLen={avg_len:5.1f} | "
            f"pl={avg_pl:.4f}  vl={avg_vl:.4f} | "
            f"buf={len(buffer):6d} | "
            f"lr={cur_lr:.2e} | "
            f"sims={cur_sims:3d} | "
            f"{elapsed:.0f}s"
        )

        metric = {
            "iter":         it,
            "policy_loss":  avg_pl,
            "value_loss":   avg_vl,
            "avg_game_len": avg_len,
            "buf_size":     len(buffer),
            "lr":           cur_lr,
            "n_sims":       cur_sims,
            "elapsed":      elapsed,
        }

        # ── Evaluation ────────────────────────────────────────────────────
        if it % cfg["eval_freq"] == 0:
            wr = eval_vs(net, device, "random",    cfg)
            wh = eval_vs(net, device, "heuristic", cfg)
            metric["wr_random"]    = wr
            metric["wr_heuristic"] = wh
            log(f"  [eval]  vs random={wr:.2f}   vs heuristic={wh:.2f}")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if it % cfg["save_freq"] == 0:
            save_checkpoint(net, optimizer, scheduler, it, cfg, out_dir,
                            tag=f"{it:04d}")
            log(f"  [ckpt]  saved checkpoint_{it:04d}.pt")

        metrics_f.write(json.dumps(metric) + "\n")

    # ── Final checkpoint ──────────────────────────────────────────────────
    save_checkpoint(net, optimizer, scheduler, stop_at, cfg, out_dir,
                    tag=f"{stop_at:04d}")
    log(sep)
    log(f"  Done.  Total time: {(time.time()-t0)/3600:.2f}h")
    log(sep)

    metrics_f.close()
    log_f.close()


if __name__ == "__main__":
    main()
