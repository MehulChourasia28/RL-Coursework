from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from Mehuls_agent.config import NetworkConfig, RuntimeConfig, SearchConfig, dataclass_to_dict, default_runtime_config
from Mehuls_agent.encoding import encode_state
from Mehuls_agent.heuristics import choose_heuristic_action, coord_to_action
from Mehuls_agent.mcts import MCTS
from Mehuls_agent.model import PolicyValueNet
from Mehuls_agent.state import GomokuState


class GomokuRLAgent:
    def __init__(self, runtime_config: RuntimeConfig | None = None):
        self.runtime_config = runtime_config or default_runtime_config()
        self.device = self._select_device(self.runtime_config.device)
        self.model: PolicyValueNet | None = None
        self.network_config = NetworkConfig(board_size=self.runtime_config.board_size)
        self.search_config = SearchConfig(simulations=self.runtime_config.simulations)
        self._load_checkpoint_if_available()

    def _select_device(self, requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_checkpoint_if_available(self) -> None:
        checkpoint_path = Path(self.runtime_config.checkpoint_path)
        if not checkpoint_path.exists():
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        network_kwargs = checkpoint.get("network_config", {})
        self.network_config = NetworkConfig(**{**dataclass_to_dict(self.network_config), **network_kwargs})
        self.search_config = replace(self.search_config, simulations=self.runtime_config.simulations)
        self.model = PolicyValueNet(**dataclass_to_dict(self.network_config)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def has_model(self) -> bool:
        return self.model is not None

    def evaluate_state(self, state: GomokuState) -> tuple[np.ndarray, float]:
        assert self.model is not None
        features = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(features)
        return policy_logits.squeeze(0).detach().cpu().numpy(), float(value.item())

    def select_action(self, board_state: np.ndarray, player: int) -> int:
        size = board_state.shape[0]
        state = GomokuState(size=size, board=board_state.astype(np.int8, copy=True), current_player=player)
        legal = state.legal_actions()
        if legal.size == 0:
            return -1

        if self.model is None and Path(self.runtime_config.checkpoint_path).exists():
            self._load_checkpoint_if_available()

        if self.model is None:
            if self.runtime_config.heuristic_fallback:
                return choose_heuristic_action(state.board.copy(), player)
            return int(legal[0])

        if self.runtime_config.use_search:
            search = MCTS(self.evaluate_state, self.search_config)
            distribution = search.search(state, training=False, move_number=state.occupied_count())
            return int(np.argmax(distribution))

        logits, _ = self.evaluate_state(state)
        logits = logits.astype(np.float64, copy=False)
        invalid = state.board.reshape(-1) != 0
        logits[invalid] = -np.inf
        return int(np.argmax(logits))

    def predict_xy(self, board_state: np.ndarray, player: int) -> tuple[int, int]:
        action = self.select_action(board_state, player)
        if action < 0:
            return -1, -1
        row, col = divmod(action, board_state.shape[0])
        return int(col), int(row)


_LOADED_AGENT: GomokuRLAgent | None = None


def load_agent(runtime_config: RuntimeConfig | None = None) -> GomokuRLAgent:
    global _LOADED_AGENT
    if _LOADED_AGENT is None or runtime_config is not None:
        _LOADED_AGENT = GomokuRLAgent(runtime_config)
    return _LOADED_AGENT


def predict_move(board_state: np.ndarray, player: int) -> tuple[int, int]:
    agent = load_agent()
    return agent.predict_xy(board_state, player)