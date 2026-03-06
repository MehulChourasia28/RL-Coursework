# Reinforcement Learning Coursework (RL-Coursework)

This repository contains coursework exercises and experiments for reinforcement learning, plus a custom Gomoku environment and UI.

## Current Project State

The project currently has two main tracks:

1. Practice RL tasks in `practice/` (bandits, DP/gridworld, and notebooks).
2. A working Gomoku implementation with:
	 - core game logic,
	 - a PyGame user interface,
	 - an RL-style environment wrapper,
	 - a human-vs-AI runner with a single `predict(board_state) -> (x, y)` hook.

The human-vs-AI setup is now ready for plugging in custom AI logic.

## Repository Structure

### Root

- `gameboard.py`
	- `GomokuLogic`: board rules, move validation, win/draw detection.
	- `GomokuGame`: PyGame visual game loop for human play.
- `gomoku_env.py`
	- RL-style wrapper around `GomokuLogic`.
	- Supports `reset()` and `step(action)` with rewards and terminal flags.
	- Optional rendering via PyGame.
- `gomoku_human_vs_ai.py`
	- Human-vs-AI game loop.
	- Contains a single function for agent logic:
		- `predict(board_state) -> (x, y)`
	- `board_state` encoding:
		- `0` empty,
		- `1` black,
		- `-1` white.
- `RLCW2526.pdf`
	- Coursework handout/specification.

### `practice/`

- `multi_arm_bandit.py`
- `dp_planner.py`
- `gridworld.py`
- `maze_MC.ipynb`
- `maze_TD.ipynb`
- `N_in_a_row.ipynb`
- `tictactoe.ipynb`
- `test.ipynb`

## Gomoku Components

### 1) Core Rules (`gameboard.py`)

`GomokuLogic` stores the board as a NumPy array and handles:

- move validity checks,
- turn updates,
- win checks (5 in a row in 4 directions),
- draw detection.

### 2) RL Environment (`gomoku_env.py`)

`GomokuEnv` provides an RL-friendly interface:

- `reset()` -> initial board state
- `step(action)` -> `(next_state, reward, done, info)`

`action` is a flattened index in `[0, 80]` for a 9x9 board.

Reward shaping currently implemented:

- `+10` for agent win,
- `-10` for invalid move,
- `-10` for opponent win,
- `0` for draw,
- `-0.1` per non-terminal step.

Opponent policy is currently random.

### 3) Human vs AI (`gomoku_human_vs_ai.py`)

This is the easiest entrypoint for custom play logic.

Edit:

```python
def predict(board_state):
		# return (x, y)
		...
```

Conventions:

- Return `(x, y)` where:
	- `x` = column index (`0..8`)
	- `y` = row index (`0..8`)
- If your agent returns an invalid move, the runner falls back to a random legal move so the game can continue.

## Setup

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install numpy pygame
```

## How To Run

### Run Human vs AI

```bash
python gomoku_human_vs_ai.py
```

By default, human plays Black (`human_player=1`).

### Run RL Environment Demo

```bash
python gomoku_env.py
```

This runs multiple episodes with random actions and optional visualization.

### Run Standalone Gomoku UI

```bash
python gameboard.py
```

This launches the two-player local PyGame interface.

## Notes and Limitations

- No trained Gomoku model is committed yet; current AI baseline is random unless you replace `predict`.
- Opponent in `gomoku_env.py` is random, which is fine for initial experimentation but limited for stronger training.
- There is no formal test suite in the repository yet.

## Suggested Next Steps

1. Implement a stronger heuristic in `predict(board_state)` (win/block-first).
2. Add a training script that learns a policy from `GomokuEnv`.
3. Save/load model weights and connect them to `predict`.
4. Add minimal tests for move validity, win detection, and reward logic.
