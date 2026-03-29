"""
Round-robin tournament between checkpoints from all study runs.

Matches:
  - Each cfg's final checkpoint (iter 250) vs every other
  - cfg1's progression: iter 50, 100, 150, 200, 250

Usage (from repo root):
    python study/eval_tournament.py
"""

import json
import os
import random
import sys
from itertools import combinations

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from gameboard import GomokuLogic
from Mehuls_agent.model import PolicyValueNet
from Mehuls_agent.mcts import MCTS

BOARD_SIZE  = 9
N_SIMS      = 200
GAMES_MATCH = 40    # games per head-to-head matchup (alternating colours)


# ─────────────────────────────────────────────────────────────────────────────
# Load agent from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(ckpt_path: str, device: torch.device, label: str):
    if not os.path.exists(ckpt_path):
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
    return {"label": label, "net": net, "device": device}


# ─────────────────────────────────────────────────────────────────────────────
# Play one match (n games, alternating colours)
# ─────────────────────────────────────────────────────────────────────────────

def play_match(agent_a: dict, agent_b: dict, n_games: int = GAMES_MATCH) -> dict:
    """Returns win/draw/loss counts from agent_a's perspective."""
    from Mehuls_agent.heuristics import heuristic_move

    device_a, net_a = agent_a["device"], agent_a["net"]
    device_b, net_b = agent_b["device"], agent_b["net"]

    mcts_a = MCTS(net_a, device_a, n_simulations=N_SIMS, dirichlet_epsilon=0.0)
    mcts_b = MCTS(net_b, device_b, n_simulations=N_SIMS, dirichlet_epsilon=0.0)

    wins_a = draws = wins_b = 0

    for gi in range(n_games):
        logic      = GomokuLogic(BOARD_SIZE)
        # Alternate who is black
        color_a    = 1 if gi % 2 == 0 else -1
        color_b    = -color_a
        cur_player = 1
        mcts_a.reset()
        mcts_b.reset()

        while not logic.game_over:
            board = logic.board.copy()
            if cur_player == color_a:
                policy = mcts_a.get_policy(board, cur_player,
                                           temperature=0.0, add_noise=False)
                valid  = (board.flatten() == 0).astype(np.float32)
                action = int(np.argmax(policy * valid))
            else:
                policy = mcts_b.get_policy(board, cur_player,
                                           temperature=0.0, add_noise=False)
                valid  = (board.flatten() == 0).astype(np.float32)
                action = int(np.argmax(policy * valid))

            r, c = divmod(action, BOARD_SIZE)
            logic.step(r, c, cur_player)
            mcts_a.advance(action)
            mcts_b.advance(action)
            cur_player = -cur_player

        if logic.winner == color_a:
            wins_a += 1
        elif logic.winner == color_b:
            wins_b += 1
        else:
            draws += 1

    return {"wins_a": wins_a, "draws": draws, "wins_b": wins_b}


# ─────────────────────────────────────────────────────────────────────────────
# Run the tournament
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = os.path.join(REPO, "study", "runs")
    plots_dir = os.path.join(REPO, "study", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Collect agents ────────────────────────────────────────────────────

    agents = []

    # Final checkpoints of each config
    for cfg_name in ["cfg1", "cfg2", "cfg3", "cfg4", "cfg5"]:
        ckpt = os.path.join(runs_dir, cfg_name, "checkpoint_0250.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(runs_dir, cfg_name, "checkpoint_latest.pt")
        ag = load_agent(ckpt, device, f"{cfg_name}@250")
        if ag:
            agents.append(ag)
            print(f"  Loaded {ag['label']}")

    # Progression of cfg1 (baseline)
    for it in [50, 100, 150, 200]:
        ckpt = os.path.join(runs_dir, "cfg1", f"checkpoint_{it:04d}.pt")
        ag = load_agent(ckpt, device, f"cfg1@{it}")
        if ag:
            agents.append(ag)
            print(f"  Loaded {ag['label']}")

    if len(agents) < 2:
        print("\nNot enough checkpoints for a tournament. Train first.")
        return

    # ── Round-robin ───────────────────────────────────────────────────────

    labels = [ag["label"] for ag in agents]
    n      = len(agents)
    matrix = np.full((n, n), np.nan)   # win-rate of row vs col

    print(f"\nRunning round-robin tournament: {n} agents, {GAMES_MATCH} games each…\n")
    results_log = []

    for i, j in combinations(range(n), 2):
        print(f"  {labels[i]}  vs  {labels[j]} … ", end="", flush=True)
        res = play_match(agents[i], agents[j])
        total = res["wins_a"] + res["draws"] + res["wins_b"]
        wr_i  = (res["wins_a"] + 0.5 * res["draws"]) / total
        wr_j  = 1.0 - wr_i
        matrix[i, j] = wr_i
        matrix[j, i] = wr_j
        print(f"  {labels[i]}: {wr_i:.2f}  |  {labels[j]}: {wr_j:.2f}")
        results_log.append({
            "a": labels[i], "b": labels[j],
            "wins_a": res["wins_a"], "draws": res["draws"], "wins_b": res["wins_b"],
            "wr_a": wr_i,
        })
        np.fill_diagonal(matrix, 0.5)   # self vs self

    # ── Win-rate table ────────────────────────────────────────────────────

    print("\n\nTournament Win-Rate Matrix (row vs col):\n")
    col_w = 12
    header = " " * 20 + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    for i, lb in enumerate(labels):
        row_str = f"{lb:<20}"
        for j in range(n):
            cell = f"{matrix[i,j]:.2f}" if not np.isnan(matrix[i,j]) else "  — "
            row_str += f"{cell:>{col_w}}"
        print(row_str)

    # ── Save results ──────────────────────────────────────────────────────

    out = {
        "labels":      labels,
        "matrix":      matrix.tolist(),
        "match_log":   results_log,
    }
    path = os.path.join(plots_dir, "tournament_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {path}")

    # ── Heatmap ───────────────────────────────────────────────────────────

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))
        masked  = np.where(np.isnan(matrix), 0.5, matrix)
        im = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="black" if 0.25 < matrix[i,j] < 0.75 else "white")

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Tournament Win-Rate Matrix  (row vs column)", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Win rate (row agent)")
        fig.tight_layout()

        plot_path = os.path.join(plots_dir, "tournament_heatmap.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved → {plot_path}")
    except ImportError:
        print("matplotlib not available – skipping heatmap")


if __name__ == "__main__":
    main()
