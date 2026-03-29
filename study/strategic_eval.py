"""
Strategic evaluation of the AlphaZero agent across 10 tactical scenarios.

Tests whether the agent can:
  1.  Win in 1 – horizontal
  2.  Win in 1 – diagonal
  3.  Block opponent's immediate win (horizontal)
  4.  Block opponent's immediate win (diagonal)
  5.  Create a fork (two simultaneous 4-in-a-row threats)
  6.  Block an opponent fork
  7.  Prefer open-4 over blocked-4
  8.  Detect a double-open-3 attack
  9.  Counter-fork (reply to a fork with a winning threat)
  10. Endgame precision (choose the unique winning line)

Usage (from repo root):
    python study/strategic_eval.py [--cfg cfg1] [--iters 50 100 250]
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Set, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from Mehuls_agent.model import PolicyValueNet, encode_state
from Mehuls_agent.mcts import MCTS

BOARD_SIZE = 9
N_SIMS_EVAL = 200


# ─────────────────────────────────────────────────────────────────────────────
# Board builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def empty_board() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def place(board: np.ndarray, stones: List[Tuple[int, int, int]]) -> np.ndarray:
    """stones: list of (row, col, player)  where player ∈ {1, -1}."""
    b = board.copy()
    for r, c, p in stones:
        b[r][c] = p
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_scenarios() -> List[dict]:
    """
    Returns a list of dicts:
        name        : short label
        description : human-readable description
        board       : 9×9 numpy board
        player      : whose turn it is (1 = black)
        correct     : set of (row, col) that are considered correct
        category    : 'offense' | 'defense' | 'fork' | 'endgame'
    """
    scenarios = []

    # ── 1. Win in 1 – horizontal ──────────────────────────────────────────
    # Black has ·1 1 1 1·  →  must play at either open end
    b = place(empty_board(), [
        (4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1),
        # some opponent stones to make board non-trivial
        (3, 4, -1), (5, 5, -1),
    ])
    scenarios.append({
        "name":        "Win1H",
        "description": "Black has open-4 horizontal; must win by extending to 5",
        "board":       b,
        "player":      1,
        "correct":     {(4, 1), (4, 6)},
        "category":    "offense",
    })

    # ── 2. Win in 1 – diagonal ────────────────────────────────────────────
    b = place(empty_board(), [
        (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1),
        (5, 6, -1), (2, 5, -1),
    ])
    scenarios.append({
        "name":        "Win1D",
        "description": "Black has open-4 diagonal (NW→SE); must win by extending",
        "board":       b,
        "player":      1,
        "correct":     {(0, 0), (5, 5)},
        "category":    "offense",
    })

    # ── 3. Block opponent win – horizontal ───────────────────────────────
    # White has ·-1 -1 -1 -1· → black must block
    b = place(empty_board(), [
        (4, 3, -1), (4, 4, -1), (4, 5, -1), (4, 6, -1),
        (3, 3,  1), (5, 4,  1),
    ])
    scenarios.append({
        "name":        "Block1H",
        "description": "White has open-4 horizontal; black must block at either end",
        "board":       b,
        "player":      1,
        "correct":     {(4, 2), (4, 7)},
        "category":    "defense",
    })

    # ── 4. Block opponent win – diagonal ─────────────────────────────────
    b = place(empty_board(), [
        (2, 2, -1), (3, 3, -1), (4, 4, -1), (5, 5, -1),
        (4, 2,  1), (2, 4,  1),
    ])
    scenarios.append({
        "name":        "Block1D",
        "description": "White has open-4 diagonal; black must block at either end",
        "board":       b,
        "player":      1,
        "correct":     {(1, 1), (6, 6)},
        "category":    "defense",
    })

    # ── 5. Create a fork ──────────────────────────────────────────────────
    # Black plays the key cell that creates two simultaneous 4-in-a-row threats.
    # Horizontal: (4,1)(4,2)(4,3)·  →  (4,4) makes hor-4 and diagonal-4
    # Diagonal:   (1,7)(2,6)(3,5)·  →  (4,4) also extends diagonal
    b = place(empty_board(), [
        (4, 1, 1), (4, 2, 1), (4, 3, 1),  # horizontal 3-in-a-row
        (1, 7, 1), (2, 6, 1), (3, 5, 1),  # diagonal 3-in-a-row (↙)
        # White has some irrelevant stones
        (7, 7, -1), (7, 8, -1),
    ])
    scenarios.append({
        "name":        "Fork",
        "description": "Playing (4,4) creates simultaneous 4-in-a-row on horizontal "
                       "and diagonal – neither can be blocked",
        "board":       b,
        "player":      1,
        "correct":     {(4, 4)},
        "category":    "fork",
    })

    # ── 6. Block an opponent fork ─────────────────────────────────────────
    # White can fork at (4,4): horizontal (4,5)(4,6)(4,7) + diagonal (7,1)(6,2)(5,3)
    # Black must either block at (4,4) or create a forcing winning threat
    b = place(empty_board(), [
        (4, 5, -1), (4, 6, -1), (4, 7, -1),   # white horizontal 3
        (7, 1, -1), (6, 2, -1), (5, 3, -1),   # white diagonal 3
        (2, 2, 1), (2, 3, 1),                  # black irrelevant
    ])
    scenarios.append({
        "name":        "BlockFork",
        "description": "White threatens fork at (4,4); black must block it",
        "board":       b,
        "player":      1,
        "correct":     {(4, 4)},
        "category":    "defense",
    })

    # ── 7. Prefer open-4 over blocked-4 ──────────────────────────────────
    # Black can extend to a blocked-4 at (4,1) or to an open-4 at (6,4).
    # The open-4 is strategically superior.
    b = place(empty_board(), [
        (4, 2, 1), (4, 3, 1), (4, 4, 1),   # closed on left by white
        (4, 0, -1),                          # blocks left end
        (6, 1, 1), (6, 2, 1), (6, 3, 1),   # open on both sides
        (3, 3, -1), (7, 6, -1),
    ])
    scenarios.append({
        "name":        "OpenVsBlocked4",
        "description": "Black can make a blocked-4 at (4,5) or an open-3→4 at (6,4); "
                       "open-4 is correct",
        "board":       b,
        "player":      1,
        "correct":     {(6, 4), (6, 0)},   # either end of the open-3
        "category":    "offense",
    })

    # ── 8. Double open-3 attack (double threat) ────────────────────────────
    # Black has two separate open-3 in-a-rows; the key move extends both to open-4.
    # This scenario tests whether the agent can create double threats.
    b = place(empty_board(), [
        (3, 3, 1), (3, 4, 1), (3, 5, 1),   # horizontal open-3
        (2, 5, 1), (4, 3, 1),               # diagonal open-3 through (3,4)
        (6, 6, -1), (7, 7, -1),
    ])
    # Playing (1,7) or (5,1): both extend one of the open-3 to open-4
    # Playing (3,6) extends horizontal; good move
    scenarios.append({
        "name":        "DoubleOpen3",
        "description": "Black has two open-3 sequences; must extend one to open-4 "
                       "to create a double threat",
        "board":       b,
        "player":      1,
        "correct":     {(3, 2), (3, 6), (1, 7), (5, 1)},
        "category":    "offense",
    })

    # ── 9. Counter-fork (attack while defending) ──────────────────────────
    # White threatens a fork; black creates a direct win threat forcing white to defend,
    # avoiding the fork without passively blocking.
    b = place(empty_board(), [
        (1, 1, -1), (2, 2, -1), (3, 3, -1),  # white diagonal 3
        (1, 5, -1), (2, 5, -1), (3, 5, -1),  # white vertical 3
        (5, 0,  1), (5, 1,  1), (5, 2,  1),  # black horizontal 3 (threat)
        (4, 8, -1),
    ])
    scenarios.append({
        "name":        "CounterFork",
        "description": "White threatens fork; black should extend own 3→4 at (5,3) or (5,-1) "
                       "creating a forcing threat, rather than passively defending",
        "board":       b,
        "player":      1,
        "correct":     {(5, 3), (5, 4)},  # extend black's attacking line
        "category":    "fork",
    })

    # ── 10. Endgame precision ─────────────────────────────────────────────
    # Dense board; exactly one winning line available for black.
    b = place(empty_board(), [
        # Cluttered middle
        (0, 0, -1), (0, 1, 1),  (0, 2, -1), (0, 3, 1),  (0, 4, -1),
        (1, 0,  1), (1, 1, -1), (1, 2,  1), (1, 3, -1), (1, 4,  1),
        (2, 0, -1), (2, 1,  1), (2, 2, -1), (2, 3,  1), (2, 4, -1),
        (3, 0,  1), (3, 1, -1), (3, 2,  1), (3, 3, -1), (3, 4,  1),
        (4, 0, -1), (4, 1,  1), (4, 2, -1), (4, 3,  1), (4, 4, -1),
        # Black has a nearly-complete line at row 8
        (8, 2, 1), (8, 3, 1), (8, 4, 1), (8, 5, 1),
        # White is far from winning in this area
        (7, 7, -1), (6, 7, -1),
    ])
    scenarios.append({
        "name":        "Endgame",
        "description": "Dense board; black has only one winning move: extending row-8 "
                       "open-4 to 5",
        "board":       b,
        "player":      1,
        "correct":     {(8, 1), (8, 6)},
        "category":    "endgame",
    })

    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Load a checkpoint and build an MCTS agent
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(ckpt_path: str, device: torch.device) -> Optional[MCTS]:
    if not os.path.exists(ckpt_path):
        print(f"  [warn] checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    net  = PolicyValueNet(
        board_size      = BOARD_SIZE,
        in_channels     = 3,
        channels        = cfg.get("channels", 64),
        num_res_blocks  = cfg.get("res_blocks", 4),
        value_fc_hidden = cfg.get("value_fc_hidden", 64),
    ).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    mcts = MCTS(net, device, n_simulations=N_SIMS_EVAL, dirichlet_epsilon=0.0)
    return mcts


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one agent on all scenarios
# ─────────────────────────────────────────────────────────────────────────────

def eval_agent(mcts: MCTS, scenarios: List[dict]) -> dict:
    results = {}
    for sc in scenarios:
        board   = sc["board"].copy()
        player  = sc["player"]
        correct = sc["correct"]

        mcts.reset()
        policy  = mcts.get_policy(board, player, temperature=0.0, add_noise=False)
        valid   = (board.flatten() == 0).astype(np.float32)
        policy  = policy * valid
        action  = int(np.argmax(policy))
        r, c    = divmod(action, BOARD_SIZE)

        # Value estimate at root
        value   = mcts.root_value

        passed  = (r, c) in correct
        results[sc["name"]] = {
            "passed":   passed,
            "move":     (r, c),
            "correct":  list(correct),
            "value":    round(float(value), 4),
            "category": sc["category"],
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print and save results
# ─────────────────────────────────────────────────────────────────────────────

def print_results(label: str, results: dict, scenarios: List[dict]):
    sep  = "─" * 72
    passed = sum(1 for v in results.values() if v["passed"])
    total  = len(results)
    print(f"\n{sep}")
    print(f"  {label}  –  Score: {passed}/{total}")
    print(sep)
    for sc in scenarios:
        name = sc["name"]
        r    = results[name]
        tick = "✓" if r["passed"] else "✗"
        print(
            f"  {tick}  {name:<18}  move={r['move']}  "
            f"correct={r['correct']}  value={r['value']:+.3f}  "
            f"[{r['category']}]"
        )
        if not r["passed"]:
            print(f"           └── {sc['description']}")
    print(sep)


def save_results(all_results: dict, out_dir: str):
    path = os.path.join(out_dir, "strategic_eval.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Generate comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_strategic(all_results: dict, scenarios: List[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [sc["name"] for sc in scenarios]
    cfgs   = list(all_results.keys())

    scores = np.zeros((len(cfgs), len(labels)))
    for ci, cfg_label in enumerate(cfgs):
        for si, sc in enumerate(scenarios):
            r = all_results[cfg_label].get(sc["name"])
            if r and isinstance(r, dict):
                scores[ci, si] = 1.0 if r["passed"] else 0.0

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(scores, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels(cfgs, fontsize=9)
    for ci in range(len(cfgs)):
        for si in range(len(labels)):
            ax.text(si, ci, "✓" if scores[ci, si] else "✗",
                    ha="center", va="center", fontsize=12,
                    color="black")
    ax.set_title("Strategic Evaluation Pass/Fail  (green=pass, red=fail)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Pass (1) / Fail (0)")
    fig.tight_layout()

    out_path = os.path.join(REPO, "study", "plots", "strategic_eval.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgs", nargs="+", default=["cfg1", "cfg2", "cfg3", "cfg4", "cfg5"],
                        help="Which configs to evaluate")
    parser.add_argument("--iters", nargs="+", type=int, default=[250],
                        help="Which checkpoint iterations to load (e.g. 50 100 250)")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenarios = build_scenarios()
    runs_dir  = os.path.join(REPO, "study", "runs")
    plots_dir = os.path.join(REPO, "study", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_results = {}  # key → results dict

    for cfg_name in args.cfgs:
        for it in args.iters:
            ckpt = os.path.join(runs_dir, cfg_name, f"checkpoint_{it:04d}.pt")
            if not os.path.exists(ckpt):
                # Try latest
                ckpt = os.path.join(runs_dir, cfg_name, "checkpoint_latest.pt")
            label = f"{cfg_name}@{it}"
            mcts  = load_agent(ckpt, device)
            if mcts is None:
                print(f"  Skipping {label} (no checkpoint)")
                continue
            results = eval_agent(mcts, scenarios)
            print_results(label, results, scenarios)
            all_results[label] = results

    if all_results:
        out_dir = os.path.join(REPO, "study", "runs")
        save_results(all_results, plots_dir)
        plot_strategic(all_results, scenarios)
    else:
        print("\nNo checkpoints found yet. Train first with run_all.sh")


if __name__ == "__main__":
    main()
