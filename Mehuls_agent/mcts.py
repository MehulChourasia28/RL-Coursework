from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from Mehuls_agent.config import SearchConfig
from Mehuls_agent.heuristics import heuristic_policy
from Mehuls_agent.state import GomokuState


Evaluator = Callable[[GomokuState], tuple[np.ndarray, float]]


@dataclass(slots=True)
class Node:
    state: GomokuState
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] | None = None

    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, evaluator: Evaluator, search_config: SearchConfig, rng: np.random.Generator | None = None):
        self.evaluator = evaluator
        self.search_config = search_config
        self.rng = rng or np.random.default_rng()

    def search(self, root_state: GomokuState, training: bool = False, move_number: int = 0) -> np.ndarray:
        root = Node(state=root_state.clone(), children={})
        priors, _ = self.evaluator(root.state)
        self._expand(root, priors, add_noise=training)

        for _ in range(self.search_config.simulations):
            node = root
            path = [node]

            while node.expanded() and not node.state.terminal:
                _, node = self._select_child(node)
                path.append(node)

            if node.state.terminal:
                value = node.state.outcome_for(node.state.current_player)
            else:
                priors, value = self.evaluator(node.state)
                self._expand(node, priors, add_noise=False)

            self._backpropagate(path, value)

        temperature = (
            self.search_config.exploration_temperature
            if training and move_number < self.search_config.temperature_moves
            else self.search_config.deterministic_temperature
        )
        return self._visit_distribution(root, temperature)

    def _select_child(self, node: Node) -> tuple[int, Node]:
        assert node.children is not None
        total_visits = math.sqrt(max(1, node.visit_count))
        best_action = -1
        best_score = -math.inf
        best_child = None
        for action, child in node.children.items():
            exploration = self.search_config.c_puct * child.prior * total_visits / (1 + child.visit_count)
            score = -child.mean_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        assert best_child is not None
        return best_action, best_child

    def _expand(self, node: Node, policy_logits: np.ndarray, add_noise: bool) -> None:
        legal_actions = node.state.legal_actions()
        node.children = {}
        if legal_actions.size == 0:
            return

        legal_logits = policy_logits[legal_actions].astype(np.float64, copy=False)
        legal_logits -= legal_logits.max()
        network_prior = np.exp(legal_logits)
        network_prior_sum = float(network_prior.sum())
        if network_prior_sum <= 0.0:
            network_prior = np.full(len(legal_actions), 1.0 / len(legal_actions), dtype=np.float64)
        else:
            network_prior /= network_prior_sum

        heuristic_prior = heuristic_policy(node.state.board, node.state.current_player)[legal_actions].astype(np.float64, copy=False)
        heuristic_sum = float(heuristic_prior.sum())
        if heuristic_sum > 0.0:
            heuristic_prior /= heuristic_sum
            blend = self.search_config.heuristic_prior_blend
            blended_prior = (1.0 - blend) * network_prior + blend * heuristic_prior
        else:
            blended_prior = network_prior

        if add_noise and len(legal_actions) > 1:
            noise = self.rng.dirichlet([self.search_config.dirichlet_alpha] * len(legal_actions))
            blended_prior = (1.0 - self.search_config.dirichlet_fraction) * blended_prior + self.search_config.dirichlet_fraction * noise

        blended_sum = float(blended_prior.sum())
        if blended_sum > 0.0:
            blended_prior /= blended_sum

        for action, prior in zip(legal_actions, blended_prior, strict=True):
            child_state = node.state.apply_action(int(action))
            node.children[int(action)] = Node(state=child_state, prior=float(prior), children={})

    def _backpropagate(self, path: list[Node], value: float) -> None:
        current_value = value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value

    def _visit_distribution(self, root: Node, temperature: float) -> np.ndarray:
        board_size = root.state.size
        distribution = np.zeros(board_size * board_size, dtype=np.float32)
        if not root.children:
            return distribution

        actions = np.array(list(root.children.keys()), dtype=np.int64)
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)

        if temperature <= 0.05:
            best_action = actions[int(np.argmax(visits))]
            distribution[best_action] = 1.0
            return distribution

        adjusted = np.power(visits + 1e-8, 1.0 / temperature)
        adjusted_sum = float(adjusted.sum())
        if adjusted_sum <= 0.0:
            distribution[actions] = 1.0 / len(actions)
            return distribution
        distribution[actions] = adjusted / adjusted_sum
        return distribution


def clone_search_config(search_config: SearchConfig, simulations: int | None = None) -> SearchConfig:
    if simulations is None:
        return replace(search_config)
    return replace(search_config, simulations=simulations)