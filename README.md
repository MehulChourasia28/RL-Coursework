# Reinforcement Learning Coursework (RL-Coursework)

This repository contains coursework exercises and experiments for reinforcement learning, plus a custom Gomoku environment and UI.

## Strong Gomoku Agent

There is now a practical RL-based Gomoku agent under `Mehuls_agent/`.

It is designed for a 15x15 board and uses:

- a policy-value CNN,
- lightweight tree search,
- heuristic bootstrapping,
- self-play training,
- checkpointed inference for the PyGame UI.

The human-vs-AI runner in `gomoku_human_vs_ai.py` now uses this agent automatically.
If no trained checkpoint is available yet, it falls back to the built-in tactical heuristic instead of random play.

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
- `gomoku_config.py`
	- Central game settings.
	- Set `BOARD_SIZE` once to change board size across Gomoku modules.

### `practice/`

- `multi_arm_bandit.py`
- `dp_planner.py`
- `gridworld.py`
- `maze_MC.ipynb`
- `maze_TD.ipynb`
- `N_in_a_row.ipynb`
- `tictactoe.ipynb`
- `test.ipynb`

### `Mehuls_agent/`

- `config.py`
	- runtime, network, search, and training configuration.
- `state.py`
	- canonical self-play game state wrapper around Gomoku rules.
- `encoding.py`
	- network input encoding and symmetry augmentation.
- `heuristics.py`
	- tactical move scoring and fallback policy.
- `model.py`
	- residual policy-value network.
- `mcts.py`
	- lightweight PUCT search.
- `train_gomoku_agent.py`
	- self-play training entrypoint.
- `inference.py`
	- checkpoint loading and move selection for the UI.

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

`action` is a flattened index in `[0, size*size - 1]`.

Reward shaping currently implemented:

- `+10` for agent win,
- `-10` for invalid move,
- `-10` for opponent win,
- `0` for draw,
- `-0.1` per non-terminal step.

Opponent policy is currently random.

### 3) Human vs AI (`gomoku_human_vs_ai.py`)

This is the easiest entrypoint for custom play logic.

It now routes moves through `Mehuls_agent.inference`.

Conventions:

- Return `(x, y)` where:
	- `x` = column index (`0..size-1`)
	- `y` = row index (`0..size-1`)
- If your agent returns an invalid move, the runner falls back to a random legal move so the game can continue.

### Board Size Configuration

To change board size globally, edit this file:

- `gomoku_config.py`

Update:

```python
BOARD_SIZE = 9
```

All Gomoku entrypoints (`gameboard.py`, `gomoku_env.py`, `gomoku_human_vs_ai.py`) read this value by default.

## Setup

Python 3.9+ is recommended.

Create the project virtual environment:

```bash
python3 -m venv .venv
```

Install base dependencies:

```bash
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install numpy pygame tqdm tensorboard
```

For the GB10 GPU used in this workspace, the stable CUDA 12.4 PyTorch wheel was not sufficient.
Use a newer CUDA 12.8 nightly build:

```bash
.venv/bin/python -m pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

You can verify CUDA is working with:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## How To Run

### Run Human vs AI

```bash
.venv/bin/python gomoku_human_vs_ai.py
```

By default, human plays Black (`human_player=1`).

The UI uses the best checkpoint at:

```bash
Mehuls_agent/checkpoints/gomoku_policy_value.pt
```

If that file does not exist yet, the UI falls back to a tactical heuristic agent.

### Train the Gomoku Agent

```bash
.venv/bin/python Mehuls_agent/train_gomoku_agent.py --device cuda --checkpoint Mehuls_agent/checkpoints/gomoku_policy_value.pt
```

Useful smaller smoke run:

```bash
.venv/bin/python Mehuls_agent/train_gomoku_agent.py --device cuda --iterations 1 --bootstrap-games 2 --self-play-games 1 --evaluation-games 2 --batch-size 8 --batches-per-iteration 2 --checkpoint Mehuls_agent/checkpoints/smoke_gomoku_policy_value.pt
```

### Run RL Environment Demo

```bash
.venv/bin/python gomoku_env.py
```

This runs multiple episodes with random actions and optional visualization.

### Run Standalone Gomoku UI

```bash
.venv/bin/python gameboard.py
```

This launches the two-player local PyGame interface.

## Notes and Limitations

- `gomoku_env.py` still uses a random opponent and is not the main training path for the strong agent.
- The practical training path is `Mehuls_agent/train_gomoku_agent.py`.
- A short smoke checkpoint can be produced quickly, but a stronger model still needs a longer CUDA training run.
- There is no formal test suite in the repository yet.

## Suggested Next Steps

1. Run a longer CUDA training job to populate `Mehuls_agent/checkpoints/gomoku_policy_value.pt`.
2. Evaluate the trained checkpoint against the tactical heuristic and human play.
3. Tune search simulations and training iterations once the first longer run completes.
4. Add fixed tactical regression positions for forced win and forced block cases.
