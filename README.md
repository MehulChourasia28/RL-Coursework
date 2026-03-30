# Alpha Zero Python Files

- `alpha_zero1.py`: Baseline minimal AlphaZero-style Gomoku trainer with self-play, MCTS, and SGD updates.
- `alpha_zero2.py`: Adds stronger training features (tree reuse, symmetry augmentation, and scheduled LR) to the minimal trainer.
- `alpha_zero2_chk.py`: Checkpoint/resume version of the minimal trainer that saves model, optimizer, scheduler, and replay buffer.
- `az_gomuku2.py`: Stable batched self-play training pipeline (v2) built from `alpha_zero2_chk.py` for faster data generation.
- `az_gomuku3.py`: Batched trainer variant with gradient clipping and curriculum MCTS simulation counts across iterations.
- `az_gomuku4.py`: v2.1 trainer with architecture/hyperparameter upgrades plus buffer prefill and robustness fixes.
- `az_gomuku5.py`: Main v5 training script with larger network, 400-sim MCTS, bigger buffer, and cosine LR schedule.
- `az_gomuku5_further.py`: Continued training experiment from v5 iter 118 using a two-phase spike-and-settle LR strategy.
- `az_gomuku5_further2.py`: Continued training experiment from v5 iter 119 with warmup self-play buffer generation and cosine refinement.
- `eval_ui.py`: Pygame UI for human-vs-agent, agent-vs-agent, and human-vs-human matches with model loading.
- `eval_vs_heuristic.py`: CLI evaluator that measures the AlphaZero agent against the heuristic bot over many games.
- `heuristics.py`: Rule-based Gomoku heuristic engine for threat detection, move scoring, and fallback move selection.
- `multi_agent_eval.py`: Batched head-to-head evaluator for two model checkpoints with configurable MCTS sims per side.
- `predict.py`: Standard stateless `predict(board, current_player)` API that returns the model's chosen move.
- `predict_test.py`: Pygame harness for testing `predict.py`, including human modes and tree-reuse vs stateless comparison.

