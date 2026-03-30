"""
AlphaZero-style Monte Carlo Tree Search for Gomoku.

Q-value convention (negamax):
  - node.W / node.N  is the average value from THAT NODE'S player's perspective
  - +1 = the player to move at this node is winning
  - PUCT selection: score = -Q(child) + c_puct * P * sqrt(N_parent) / (1 + N_child)
    (negate Q because the child's player is the opponent of the current player)
  - Backup: flip sign at each edge going up the tree

Tree reuse:
  - Call reset() at the start of each new game.
  - Call advance(action) after every move is played.
    The child subtree for that action is kept; everything else is discarded.
    This means subsequent get_policy() calls build on top of prior simulations
    rather than starting from scratch — effectively multiplying search depth
    by ~game_length for free.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .model import PolicyValueNet, encode_state

BOARD_SIZE = 9


# ---------------------------------------------------------------------------
# Fast inline win detection
# ---------------------------------------------------------------------------

def _check_winner(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """Check if placing player at (row, col) has created a 5-in-a-row."""
    size = board.shape[0]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            for step in range(1, 5):
                r = row + sign * dr * step
                c = col + sign * dc * step
                if 0 <= r < size and 0 <= c < size and board[r, c] == player:
                    count += 1
                else:
                    break
        if count >= 5:
            return True
    return False


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class Node:
    """
    Each node represents a board state.
    N, W, Q are from the perspective of the player TO MOVE at this node.
    """
    __slots__ = ("parent", "action", "children", "N", "W", "Q", "P")

    def __init__(self, parent: Optional["Node"], action: Optional[int], prior: float):
        self.parent  = parent
        self.action  = action          # flat action that led here from parent
        self.children: Dict[int, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior                 # prior probability from policy head

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self, c_puct: float) -> Tuple[int, "Node"]:
        """PUCT: score = -Q(child) + c_puct * P * sqrt(N) / (1 + N_child)"""
        sqrt_N = math.sqrt(max(self.N, 1))
        best_score = -float("inf")
        best_action, best_child = -1, None
        for action, child in self.children.items():
            score = -child.Q + c_puct * child.P * sqrt_N / (1 + child.N)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, priors: np.ndarray, valid_mask: np.ndarray):
        """Create children for every legal move, using network priors."""
        p = priors * valid_mask
        s = p.sum()
        p = p / s if s > 1e-10 else valid_mask / valid_mask.sum()
        for a in np.where(valid_mask)[0]:
            self.children[int(a)] = Node(self, int(a), float(p[a]))

    def backup(self, value: float):
        """
        Propagate value up the tree.
        value is from THIS node's player's perspective.
        Sign flips at each edge (opponent's perspective).
        """
        node = self
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            node.Q  = node.W / node.N
            v = -v
            node = node.parent


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(
        self,
        net: PolicyValueNet,
        device: torch.device,
        n_simulations: int = 400,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.net       = net
        self.device    = device
        self.n_sims    = n_simulations
        self.c_puct    = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_eps   = dirichlet_epsilon

        self._root: Optional[Node] = None   # persistent root for tree reuse

    # ------------------------------------------------------------------
    # Tree lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Call at the start of each new game to discard any previous tree."""
        self._root = None

    def advance(self, action: int):
        """
        After a move is played (by either player), reuse the child subtree.

        If the action was already explored, the child node becomes the new root
        and inherits all visit statistics accumulated so far.
        If the action was never explored (shouldn't happen in normal play),
        the tree is discarded and rebuilt from scratch on the next call.
        """
        if self._root is not None and action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None   # detach so old nodes can be GC'd
        else:
            self._root = None

    # ------------------------------------------------------------------
    # Neural net inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer(self, board: np.ndarray, player: int) -> Tuple[np.ndarray, float]:
        """Return (policy_probs, value) from the network."""
        state = encode_state(board, player)
        x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        log_p, v = self.net(x)
        probs = log_p.exp().squeeze(0).cpu().numpy()
        return probs, float(v.item())

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def get_policy(
        self,
        board: np.ndarray,
        player: int,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS and return a move probability distribution.
        Reuses the existing root subtree if advance() was called after the last move.

        Returns:
            policy: (board_size^2,) float32 array
        """
        size = board.shape[0]

        # Initialise root if needed (fresh game or unseen opponent move)
        if self._root is None:
            self._root = Node(None, None, 1.0)
            valid_mask = (board.flatten() == 0).astype(np.float32)
            priors, _ = self._infer(board, player)
            self._root.expand(priors, valid_mask)
            self._root.N = 1   # virtual root visit so sqrt(N) is well-defined

        root = self._root

        # Dirichlet noise on root priors for training exploration
        if add_noise and root.children:
            actions = list(root.children.keys())
            noise   = np.random.dirichlet([self.dir_alpha] * len(actions))
            for a, n in zip(actions, noise):
                c = root.children[a]
                c.P = (1.0 - self.dir_eps) * c.P + self.dir_eps * n

        # ── Simulations ──────────────────────────────────────────────────
        for _ in range(self.n_sims):
            node       = root
            sim_board  = board.copy()
            sim_player = player
            done       = False

            # Selection: walk down the tree
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                r, c = divmod(action, size)
                sim_board[r, c] = sim_player

                if _check_winner(sim_board, r, c, sim_player):
                    # sim_player just won.
                    # node's player is -sim_player (about to move but already lost).
                    node.backup(-1.0)
                    done = True
                    break

                sim_player = -sim_player

            if done:
                continue

            # Evaluation at leaf
            valid = (sim_board.flatten() == 0).astype(np.float32)
            if valid.sum() == 0:
                node.backup(0.0)
                continue

            priors, value = self._infer(sim_board, sim_player)
            node.expand(priors, valid)
            node.backup(value)

        # ── Build policy from visit counts ───────────────────────────────
        visits = np.array(
            [root.children[a].N if a in root.children else 0
             for a in range(size * size)],
            dtype=np.float32,
        )

        if temperature == 0.0:
            policy = np.zeros(size * size, dtype=np.float32)
            policy[int(np.argmax(visits))] = 1.0
        else:
            visits_t = visits ** (1.0 / temperature)
            s = visits_t.sum()
            policy = visits_t / s if s > 0 else visits / (visits.sum() + 1e-10)

        return policy

    @property
    def root_value(self) -> float:
        """MCTS-backed value estimate at the current root (current player's perspective)."""
        if self._root is None or self._root.N == 0:
            return 0.0
        return float(self._root.Q)
