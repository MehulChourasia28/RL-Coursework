from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from Mehuls_agent.config import NetworkConfig, SearchConfig, TrainingConfig, dataclass_to_dict, default_training_config
from Mehuls_agent.encoding import augment_example, encode_state
from Mehuls_agent.heuristics import choose_heuristic_action, heuristic_policy
from Mehuls_agent.mcts import MCTS, clone_search_config
from Mehuls_agent.model import PolicyValueNet
from Mehuls_agent.state import GomokuState, random_opening_actions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfPlayTrainer:
    def __init__(self, training_config: TrainingConfig, device: torch.device):
        self.training_config = training_config
        self.device = device
        self.network_config = NetworkConfig(board_size=15)
        self.search_config = SearchConfig()
        self.model = PolicyValueNet(**dataclass_to_dict(self.network_config)).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.policy_loss = nn.KLDivLoss(reduction="batchmean")
        self.value_loss = nn.MSELoss()
        self.replay_buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=self.training_config.replay_buffer_size)
        self.rng = np.random.default_rng(self.training_config.seed)
        self.best_heuristic_score = -1.0

    def evaluate_state(self, state: GomokuState) -> tuple[np.ndarray, float]:
        self.model.eval()
        features = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(features)
        return policy_logits.squeeze(0).cpu().numpy(), float(value.item())

    def build_search(self, simulations: int | None = None) -> MCTS:
        return MCTS(self.evaluate_state, clone_search_config(self.search_config, simulations), self.rng)

    def bootstrap_with_heuristics(self) -> None:
        examples = []
        for _ in tqdm(range(self.training_config.bootstrap_games), desc="Bootstrap games"):
            state = GomokuState.new_game(self.network_config.board_size)
            opening_count = int(self.rng.integers(0, 3))
            for action in random_opening_actions(state, self.rng, opening_count):
                state = state.apply_action(action)
                if state.terminal:
                    break

            trace: list[tuple[np.ndarray, np.ndarray, int]] = []
            while not state.terminal and state.occupied_count() < self.training_config.max_moves_per_game:
                policy = heuristic_policy(state.board.copy(), state.current_player)
                action = int(np.argmax(policy))
                trace.append((encode_state(state), policy.astype(np.float32), state.current_player))
                state = state.apply_action(action)

            for features, policy, player in trace:
                value = state.outcome_for(player)
                examples.append((features, policy, value))
        self._append_examples(examples)
        self.optimize(steps=max(40, self.training_config.batches_per_iteration // 4))

    def self_play_iteration(self) -> None:
        search = self.build_search()
        collected = []
        for _ in tqdm(range(self.training_config.self_play_games_per_iteration), desc="Self-play games"):
            state = GomokuState.new_game(self.network_config.board_size)
            opening_count = int(self.rng.integers(0, 3))
            for action in random_opening_actions(state, self.rng, opening_count):
                state = state.apply_action(action)
                if state.terminal:
                    break

            trace: list[tuple[np.ndarray, np.ndarray, int]] = []
            while not state.terminal and state.occupied_count() < self.training_config.max_moves_per_game:
                distribution = search.search(state, training=True, move_number=state.occupied_count())
                action = int(self.rng.choice(np.arange(len(distribution)), p=distribution))
                trace.append((encode_state(state), distribution.astype(np.float32), state.current_player))
                state = state.apply_action(action)

            for features, policy, player in trace:
                value = state.outcome_for(player)
                collected.append((features, policy, value))
        self._append_examples(collected)

    def _append_examples(self, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        for features, policy, value in examples:
            for augmented_features, augmented_policy in augment_example(features, policy, self.network_config.board_size):
                self.replay_buffer.append((augmented_features, augmented_policy, float(value)))

    def optimize(self, steps: int | None = None) -> None:
        if len(self.replay_buffer) < self.training_config.batch_size:
            return

        steps = steps or self.training_config.batches_per_iteration
        self.model.train()
        for _ in tqdm(range(steps), desc="Optimizing", leave=False):
            batch = random.sample(self.replay_buffer, self.training_config.batch_size)
            features = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device)
            target_policy = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=self.device)
            target_value = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)

            pred_policy, pred_value = self.model(features)
            log_probs = torch.log_softmax(pred_policy, dim=1)
            policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
            value_loss = self.value_loss(pred_value, target_value)
            loss = policy_loss + self.training_config.value_loss_weight * value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm)
            self.optimizer.step()

    def evaluate_against_heuristic(self, games: int) -> float:
        self.model.eval()
        wins = 0.0
        for game_index in range(games):
            state = GomokuState.new_game(self.network_config.board_size)
            learned_player = 1 if game_index % 2 == 0 else -1
            search = self.build_search(simulations=max(24, self.search_config.simulations // 2))

            while not state.terminal:
                if state.current_player == learned_player:
                    distribution = search.search(state, training=False, move_number=state.occupied_count())
                    action = int(np.argmax(distribution))
                else:
                    action = choose_heuristic_action(state.board.copy(), state.current_player)
                state = state.apply_action(action)

            result = state.outcome_for(learned_player)
            if result > 0:
                wins += 1.0
            elif result == 0:
                wins += 0.5
        return wins / games

    def save_checkpoint(self, path: str, heuristic_score: float) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "network_config": dataclass_to_dict(self.network_config),
                "search_config": dataclass_to_dict(self.search_config),
                "training_config": dataclass_to_dict(self.training_config),
                "heuristic_score": heuristic_score,
            },
            checkpoint_path,
        )

    def train(self) -> None:
        self.bootstrap_with_heuristics()
        for iteration in range(1, self.training_config.iterations + 1):
            print(f"\nIteration {iteration}/{self.training_config.iterations}")
            self.self_play_iteration()
            self.optimize()
            heuristic_score = self.evaluate_against_heuristic(self.training_config.evaluation_games)
            print(f"Heuristic evaluation score: {heuristic_score:.3f}")
            if heuristic_score >= self.best_heuristic_score:
                self.best_heuristic_score = heuristic_score
                self.save_checkpoint(self.training_config.checkpoint_path, heuristic_score)
                print(f"Saved checkpoint to {self.training_config.checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a practical Gomoku policy-value agent")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--bootstrap-games", type=int, default=None)
    parser.add_argument("--self-play-games", type=int, default=None)
    parser.add_argument("--evaluation-games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches-per-iteration", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config = default_training_config()
    if args.iterations is not None:
        training_config = replace(training_config, iterations=args.iterations)
    if args.bootstrap_games is not None:
        training_config = replace(training_config, bootstrap_games=args.bootstrap_games)
    if args.self_play_games is not None:
        training_config = replace(training_config, self_play_games_per_iteration=args.self_play_games)
    if args.evaluation_games is not None:
        training_config = replace(training_config, evaluation_games=args.evaluation_games)
    if args.batch_size is not None:
        training_config = replace(training_config, batch_size=args.batch_size)
    if args.batches_per_iteration is not None:
        training_config = replace(training_config, batches_per_iteration=args.batches_per_iteration)
    if args.checkpoint is not None:
        training_config = replace(training_config, checkpoint_path=args.checkpoint)

    set_seed(training_config.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")
    trainer = SelfPlayTrainer(training_config, device)
    trainer.train()


if __name__ == "__main__":
    main()