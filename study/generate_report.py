"""
Populate report.tex with real numbers from training metrics and eval results.

Reads:
  study/plots/summary_table.json      (from analyze_results.py)
  study/plots/strategic_eval.json     (from strategic_eval.py)
  study/plots/tournament_results.json (from eval_tournament.py)
  study/runs/*/metrics.jsonl

Writes:
  study/report_final.tex   (ready to compile with pdflatex)

Usage (from repo root):
    python study/generate_report.py
"""

import json
import os
import re
import sys
from pathlib import Path

REPO      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY     = os.path.join(REPO, "study")
PLOTS_DIR = os.path.join(STUDY, "plots")
RUNS_DIR  = os.path.join(STUDY, "runs")
TEMPLATE  = os.path.join(STUDY, "report.tex")
OUTPUT    = os.path.join(STUDY, "report_final.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Load auxiliary data
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def load_metrics(cfg_name: str):
    path = os.path.join(RUNS_DIR, cfg_name, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Extract summary statistics from metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_final_eval(rows):
    wr_r = wr_h = None
    for r in reversed(rows):
        if wr_r is None and "wr_random" in r:
            wr_r = r["wr_random"]
        if wr_h is None and "wr_heuristic" in r:
            wr_h = r["wr_heuristic"]
        if wr_r is not None and wr_h is not None:
            break
    return wr_r, wr_h


def make_table_row(cfg_name: str, rows, label: str) -> str:
    """Return one LaTeX table row for the results table."""
    if not rows:
        return f"    {label} & -- & -- & -- & -- & -- \\\\\n"
    last    = rows[-1]
    pl      = last.get("policy_loss", float("nan"))
    vl      = last.get("value_loss",  float("nan"))
    t_hrs   = last.get("elapsed", 0) / 3600
    wr_r, wr_h = get_final_eval(rows)
    wr_r_s  = f"{wr_r:.2f}" if wr_r is not None else "--"
    wr_h_s  = f"{wr_h:.2f}" if wr_h is not None else "--"
    return (
        f"    {label} & {pl:.3f} & {vl:.4f} & "
        f"{wr_r_s} & {wr_h_s} & {t_hrs:.1f}h \\\\\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fill the template
# ─────────────────────────────────────────────────────────────────────────────

def fill(template: str, data: dict) -> str:
    """Replace all  %%KEY%%  placeholders in the template."""
    for key, value in data.items():
        template = template.replace(f"%%{key}%%", str(value))
    return template


def main():
    # ── Load all data ─────────────────────────────────────────────────────
    summary  = load_json(os.path.join(PLOTS_DIR, "summary_table.json"),      {})
    strat    = load_json(os.path.join(PLOTS_DIR, "strategic_eval.json"),     {})
    tourney  = load_json(os.path.join(PLOTS_DIR, "tournament_results.json"), {})

    cfg_names = ["cfg1", "cfg2", "cfg3", "cfg4", "cfg5"]
    all_rows  = {n: load_metrics(n) for n in cfg_names}

    # ── Build placeholder values ──────────────────────────────────────────
    replacements = {}

    # Results table rows
    cfg_labels = {
        "cfg1": r"Cfg1 (5-blk, MS[50,75], $\gamma$=0.1)",
        "cfg2": r"Cfg2 (5-blk, MS[50,80,105,130], $\gamma$=0.5)",
        "cfg3": r"Cfg3 (5-blk, prog-sims, clip)",
        "cfg4": r"Cfg4 (7-blk 96ch, cosine)",
        "cfg5": r"Cfg5 (10-blk 128ch, cosine)",
    }

    table_body = ""
    for n in cfg_names:
        table_body += make_table_row(n, all_rows[n], cfg_labels[n])
    replacements["RESULTS_TABLE"] = table_body

    # Best config by heuristic win rate
    best = "--"
    best_wr = -1
    for n in cfg_names:
        rows = all_rows[n]
        _, wh = get_final_eval(rows)
        if wh is not None and wh > best_wr:
            best_wr = wh
            best    = f"{n} ({wh:.2f})"
    replacements["BEST_CFG"]          = best
    replacements["BEST_WR_HEURISTIC"] = f"{best_wr:.2f}" if best_wr >= 0 else "--"

    # Strategic eval summary
    if strat:
        for label, results in strat.items():
            if isinstance(results, dict):
                passed = sum(1 for v in results.values()
                             if isinstance(v, dict) and v.get("passed"))
                total  = len(results)
                replacements[f"STRAT_{label.upper().replace('@','_').replace('-','_')}"] = \
                    f"{passed}/{total}"

        # Overall best strategic agent
        best_strat = "--"
        best_score = -1
        for label, results in strat.items():
            if isinstance(results, dict):
                score = sum(1 for v in results.values()
                            if isinstance(v, dict) and v.get("passed"))
                if score > best_score:
                    best_score = score
                    best_strat = f"{label} ({score}/{len(results)})"
        replacements["BEST_STRATEGIC"] = best_strat
    else:
        replacements["BEST_STRATEGIC"] = "(evaluation pending)"

    # Tournament winner
    if tourney and tourney.get("labels") and tourney.get("matrix"):
        import numpy as np
        mat    = np.array(tourney["matrix"])
        labels = tourney["labels"]
        np.fill_diagonal(mat, np.nan)
        mean_wr = np.nanmean(mat, axis=1)
        winner_idx = int(np.nanargmax(mean_wr))
        replacements["TOURNAMENT_WINNER"] = \
            f"{labels[winner_idx]} (avg win-rate {mean_wr[winner_idx]:.2f})"
    else:
        replacements["TOURNAMENT_WINNER"] = "(tournament pending)"

    # ── Load template and fill ────────────────────────────────────────────
    if not os.path.exists(TEMPLATE):
        print(f"Template not found: {TEMPLATE}")
        sys.exit(1)

    with open(TEMPLATE) as f:
        tex = f.read()

    tex = fill(tex, replacements)

    with open(OUTPUT, "w") as f:
        f.write(tex)

    print(f"Generated → {OUTPUT}")
    print("\nTo compile:")
    print(f"  cd study && pdflatex report_final.tex")


if __name__ == "__main__":
    main()
