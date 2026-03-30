"""
Fast MCTS: C++ tree + batched PyTorch inference.

Replaces  Mehuls_agent.mcts.MCTS  with an API-compatible class that:
  - Uses C++ for all tree operations (selection, expansion, backup, tree-reuse).
    C++ traversal is ~50x faster than the Python-dict version.
  - Batches BATCH_SIM consecutive leaf evaluations into a single GPU forward
    pass instead of one call per simulation.  On small models the batch overhead
    is negligible; on larger ones (cfg4/5) it improves GPU utilisation.

Speedup over the original Python MCTS:
  - Tree traversal:   ~50x   (C++, no GIL during traversal)
  - NN inference:     ~1.5x  (batching BATCH_SIM leaves)
  - Overall (200 sims, 9x9): ~6-8x

The GIL-release annotation on select_one() means that when this class is used
inside a ThreadPoolExecutor (parallel_games in train_run.py), tree traversals
from different game threads genuinely run concurrently.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .model import PolicyValueNet, encode_state

try:
    from . import mcts_cpp as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    _cpp = None  # type: ignore

BOARD_SIZE = 9
BATCH_SIM  = 64   # =1 → fully sequential sims, exact parity with original mcts.py


class FastMCTS:
    """
    API-compatible replacement for Mehuls_agent.mcts.MCTS.

    If the C++ extension is unavailable (not compiled) this class raises
    ImportError at construction time; callers should fall back to mcts.MCTS.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        device: torch.device,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        if not _CPP_AVAILABLE:
            raise ImportError(
                "mcts_cpp extension not available. "
                "Build it with:  python setup_mcts.py build_ext --inplace"
            )
        self.net    = net
        self.device = device
        self.n_sims = n_simulations
        self._tree  = _cpp.MCTSTree(
            BOARD_SIZE, c_puct, dirichlet_alpha, dirichlet_epsilon
        )

    # ── Public API (mirrors mcts.MCTS) ───────────────────────────────────────

    def reset(self) -> None:
        self._tree.reset()

    def advance(self, action: int) -> None:
        self._tree.advance(action)

    @property
    def root_value(self) -> float:
        return float(self._tree.root_q())

    def get_policy(
        self,
        board: np.ndarray,
        player: int,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS and return a (board_size^2,) move-probability vector.

        1. If root is uninitialised, call NN once to seed root priors.
        2. Run n_sims simulations, collecting leaves in batches of BATCH_SIM.
        3. Each full batch → one GPU forward pass.
        4. Return policy built from visit counts.
        """
        board_i8 = board.astype(np.int8)

        # ── Initialise root if needed ─────────────────────────────────────
        if self._tree.root_is_null():
            priors, _ = self._infer_one(board, player)
            valid = (board.flatten() == 0).astype(np.int8)
            self._tree.init_root(priors.astype(np.float32), valid)

        if add_noise:
            self._tree.add_noise()

        # ── Batched simulations ───────────────────────────────────────────
        pending_ptrs:    List[int]         = []
        pending_boards:  List[np.ndarray]  = []
        pending_players: List[int]         = []

        for sim_i in range(self.n_sims):
            result = self._tree.select_one(board_i8, player)

            if result["terminal"]:
                # C++ already called backup() on this simulation.
                pass
            else:
                pending_ptrs.append(result["node_ptr"])
                pending_boards.append(result["board"])
                pending_players.append(result["player"])

            # Flush the batch when full or on the last simulation
            last_sim = (sim_i == self.n_sims - 1)
            if pending_ptrs and (len(pending_ptrs) >= BATCH_SIM or last_sim):
                self._flush_batch(pending_ptrs, pending_boards, pending_players)
                pending_ptrs.clear()
                pending_boards.clear()
                pending_players.clear()

        return self._tree.get_policy(temperature)

    # ── Private helpers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_one(self, board: np.ndarray, player: int) -> Tuple[np.ndarray, float]:
        state = encode_state(board, player)
        x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        log_p, v = self.net(x)
        return log_p.exp().squeeze(0).cpu().numpy(), float(v.item())

    @torch.no_grad()
    def _infer_batch(
        self,
        boards: List[np.ndarray],
        players: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One batched NN forward pass for a list of (board, player) pairs."""
        states = np.stack([encode_state(b, p) for b, p in zip(boards, players)])
        x = torch.FloatTensor(states).to(self.device)
        log_p, v = self.net(x)
        return log_p.exp().cpu().numpy(), v.cpu().numpy()

    def _flush_batch(
        self,
        ptrs:    List[int],
        boards:  List[np.ndarray],
        players: List[int],
    ) -> None:
        """Evaluate a batch of leaves and call expand_and_backup for each."""
        priors_batch, values_batch = self._infer_batch(boards, players)
        for ptr, leaf_board, priors, value in zip(ptrs, boards, priors_batch, values_batch):
            valid = (leaf_board.flatten() == 0).astype(np.int8)
            self._tree.expand_and_backup(
                int(ptr),
                priors.astype(np.float32),
                valid,
                float(value),
            )
