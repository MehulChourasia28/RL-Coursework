"""
Analyze training results from all study runs and generate plots.

Usage (from repo root):
    python study/analyze_results.py

Reads:  study/runs/*/metrics.jsonl
Writes: study/plots/*.png  +  study/plots/summary_table.json
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS   = os.path.join(REPO, "study", "runs")
PLOTS  = os.path.join(REPO, "study", "plots")
os.makedirs(PLOTS, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     11,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "lines.linewidth": 1.8,
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

CFG_LABELS = {
    "cfg1": "Cfg1 (5-blk, MS[50,75]γ=0.1)",
    "cfg2": "Cfg2 (5-blk, MS[50,80,105,130]γ=0.5)",
    "cfg3": "Cfg3 (5-blk, prog-sims, clip)",
    "cfg4": "Cfg4 (7-blk 96ch, prog-sims)",
    "cfg5": "Cfg5 (10-blk 128ch, 400 sims)",
}


# ── Data loading ───────────────────────────────────────────────────────────

def load_run(name: str):
    path = os.path.join(RUNS, name, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get(rows, key):
    """Extract a key across all rows; returns (iters, values) with NaN for missing."""
    iters, vals = [], []
    for r in rows:
        if key in r:
            iters.append(r["iter"])
            vals.append(r[key])
    return np.array(iters), np.array(vals)


def get_eval(rows, key):
    """Extract an eval key (only present every eval_freq iters)."""
    iters, vals = [], []
    for r in rows:
        if key in r:
            iters.append(r["iter"])
            vals.append(r[key])
    return np.array(iters), np.array(vals)


def smooth(y, w=5):
    """Running mean smoothing."""
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


# ── Plot helpers ───────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Individual metric plots ────────────────────────────────────────────────

def plot_metric(all_data, metric_key, ylabel, title, filename,
                eval_key=False, smooth_w=5, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, rows) in enumerate(all_data.items()):
        if rows is None:
            continue
        color = PALETTE[i % len(PALETTE)]
        label = CFG_LABELS.get(name, name)
        if eval_key:
            xs, ys = get_eval(rows, metric_key)
        else:
            xs, ys = get(rows, metric_key)
        if len(xs) == 0:
            continue
        # Plot raw (faint) + smoothed
        ax.plot(xs, ys, alpha=0.25, color=color, lw=1.0)
        w = min(smooth_w, max(1, len(ys) // 4))
        ys_s = smooth(ys, w)
        xs_s = xs[w - 1:]
        ax.plot(xs_s, ys_s, color=color, label=label)

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, filename)


# ── LR schedule comparison ────────────────────────────────────────────────

def plot_lr_schedules(all_data):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, rows) in enumerate(all_data.items()):
        if rows is None:
            continue
        xs, ys = get(rows, "lr")
        if len(xs) == 0:
            continue
        ax.semilogy(xs, ys, color=PALETTE[i % len(PALETTE)],
                    label=CFG_LABELS.get(name, name))
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Learning Rate Schedules Across Configurations")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "lr_schedules.png")


# ── MCTS sim schedule (for cfg3 & cfg4) ────────────────────────────────────

def plot_sim_schedule(all_data):
    prog_cfgs = {k: v for k, v in all_data.items()
                 if v and any("n_sims" in r for r in v) and k in ("cfg3", "cfg4")}
    if not prog_cfgs:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, (name, rows) in enumerate(prog_cfgs.items()):
        xs = [r["iter"] for r in rows if "n_sims" in r]
        ys = [r["n_sims"] for r in rows if "n_sims" in r]
        ax.step(xs, ys, color=PALETTE[list(all_data).index(name) % len(PALETTE)],
                label=CFG_LABELS.get(name, name), where="post")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("MCTS Simulations / Move")
    ax.set_title("Progressive MCTS Simulation Schedule (Cfg3 & Cfg4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "mcts_sim_schedule.png")


# ── Final performance bar chart ────────────────────────────────────────────

def plot_final_performance(all_data):
    names, wr_r, wr_h = [], [], []
    for name, rows in all_data.items():
        if rows is None:
            continue
        _, yr = get_eval(rows, "wr_random")
        _, yh = get_eval(rows, "wr_heuristic")
        if len(yr) == 0:
            continue
        names.append(CFG_LABELS.get(name, name))
        wr_r.append(float(yr[-1]))
        wr_h.append(float(yh[-1]))

    if not names:
        print("  [warn] No eval data available yet for bar chart – skipping.")
        return

    x      = np.arange(len(names))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, wr_r, width, label="vs Random",    color="#1f77b4")
    bars2 = ax.bar(x + width / 2, wr_h, width, label="vs Heuristic", color="#ff7f0e")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Win Rate (final evaluation)")
    ax.set_title("Final Agent Performance – All Configurations")
    ax.legend()
    fig.tight_layout()
    save(fig, "final_performance.png")


# ── Avg game length heatmap (iter vs config) ─────────────────────────────

def plot_game_length(all_data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, rows) in enumerate(all_data.items()):
        if rows is None:
            continue
        xs, ys = get(rows, "avg_game_len")
        if len(xs) == 0:
            continue
        ys_s = smooth(ys, 7)
        xs_s = xs[6:]
        ax.plot(xs_s, ys_s, color=PALETTE[i % len(PALETTE)],
                label=CFG_LABELS.get(name, name))
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Game Length (moves)")
    ax.set_title("Self-Play Game Length Over Training\n"
                 "(longer games ≈ more balanced, strategic play)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "game_length.png")


# ── Combined 2×2 overview figure ─────────────────────────────────────────

def plot_overview(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    specs = [
        ("policy_loss",   "Policy Loss",             "Policy Loss (cross-entropy)",        False, None),
        ("value_loss",    "Value Loss",               "Value Loss (MSE)",                   False, (0, 0.5)),
        ("wr_random",     "Win Rate vs Random",       "Win Rate vs Random Opponent",        True,  (0, 1.1)),
        ("wr_heuristic",  "Win Rate vs Heuristic",    "Win Rate vs Heuristic Opponent",     True,  (0, 1.1)),
    ]

    for ax, (key, ylabel, title, is_eval, ylim) in zip(axes, specs):
        for i, (name, rows) in enumerate(all_data.items()):
            if rows is None:
                continue
            color = PALETTE[i % len(PALETTE)]
            label = CFG_LABELS.get(name, name)
            if is_eval:
                xs, ys = get_eval(rows, key)
            else:
                xs, ys = get(rows, key)
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, alpha=0.2, color=color, lw=0.8)
            w  = min(7, max(1, len(ys) // 4))
            ax.plot(xs[w - 1:], smooth(ys, w), color=color, label=label, lw=1.5)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("AlphaZero Gomoku – Training Overview (All Configs)", fontsize=13)
    fig.tight_layout()
    save(fig, "overview.png")


# ── Summary table ──────────────────────────────────────────────────────────

def build_summary(all_data):
    summary = {}
    for name, rows in all_data.items():
        if rows is None:
            summary[name] = {"status": "not_started"}
            continue
        last = rows[-1]
        _, yr = get_eval(rows, "wr_random")
        _, yh = get_eval(rows, "wr_heuristic")
        summary[name] = {
            "status":         "complete" if last["iter"] >= 250 else "partial",
            "iters_done":     last["iter"],
            "final_pl":       last.get("policy_loss"),
            "final_vl":       last.get("value_loss"),
            "final_wr_random":    float(yr[-1]) if len(yr) > 0 else None,
            "final_wr_heuristic": float(yh[-1]) if len(yh) > 0 else None,
            "elapsed_hours":  last.get("elapsed", 0) / 3600,
        }
    path = os.path.join(PLOTS, "summary_table.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved → {path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Analyzing study runs")
    print("=" * 60)

    cfg_names = ["cfg1", "cfg2", "cfg3", "cfg4", "cfg5"]
    all_data  = {}
    for name in cfg_names:
        rows = load_run(name)
        all_data[name] = rows
        n = len(rows) if rows else 0
        print(f"  {name}: {n} iterations loaded")

    if all(v is None for v in all_data.values()):
        print("\n  No data yet – start training with run_all.sh first.")
        return

    print("\nGenerating plots…")

    plot_metric(all_data, "policy_loss", "Policy Loss (cross-entropy)",
                "Policy Loss Across Configurations", "policy_loss.png", smooth_w=7)

    plot_metric(all_data, "value_loss", "Value Loss (MSE)",
                "Value Loss Across Configurations", "value_loss.png",
                smooth_w=7, ylim=(0, 0.5))

    plot_metric(all_data, "wr_random", "Win Rate vs Random",
                "Win Rate vs Random Opponent", "wr_random.png",
                eval_key=True, smooth_w=3, ylim=(0, 1.05))

    plot_metric(all_data, "wr_heuristic", "Win Rate vs Heuristic",
                "Win Rate vs Heuristic Opponent", "wr_heuristic.png",
                eval_key=True, smooth_w=3, ylim=(0, 1.05))

    plot_game_length(all_data)
    plot_lr_schedules(all_data)
    plot_sim_schedule(all_data)
    plot_final_performance(all_data)
    plot_overview(all_data)

    print("\nBuilding summary table…")
    summary = build_summary(all_data)
    print("\nSummary:")
    for name, info in summary.items():
        print(f"  {name}: {info}")

    print("\nDone. Plots saved to study/plots/")


if __name__ == "__main__":
    main()
