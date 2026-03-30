"""
AlphaZero — 9×9 Gomoku, minimal reference implementation.

Sections:
  1. Config
  2. Game
  3. Network
  4. MCTS
  5. Self-play
  6. Training
  7. Main loop
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

# ── 1. Config ─────────────────────────────────────────────────────────────────
# Every number here has a paper citation in the hyperparameter notes below.

BOARD         = 9
WIN           = 5
N_PLANES      = 3          # current stones | opponent stones | turn indicator
N_BLOCKS      = 5          # residual blocks — AGZ uses 20/40 for 19×19 Go;
                           # 5 is appropriate for 9×9 (much smaller state space)
N_FILTERS     = 64         # channel width — AGZ uses 256; 64 is sufficient here

N_SIMS        = 200        # MCTS simulations per move
C_PUCT        = 1.5        # exploration constant from AGZ paper §A (search params)
DIR_ALPHA     = 0.3        # Dirichlet α — AGZ uses 0.3 for Go, scales with ~10/moves
DIR_EPS       = 0.25       # noise weight at root — straight from AGZ paper
TEMP_MOVES    = 10         # play stochastically for first N moves, then greedy
                           # AGZ uses 30 for 19×19; scaled ≈ (9²/19²)×30 ≈ 7, rounded to 10

BUFFER_SIZE   = 50_000     # replay buffer capacity in (state, π, z) tuples
BATCH_SIZE    = 256        # mini-batch size — AGZ uses 2048; 256 fine for 9×9
LR            = 1e-2       # SGD lr — AGZ starts at 0.2, decays; 1e-2 is safe default
WEIGHT_DECAY  = 1e-4       # L2 regularisation weight — from AGZ paper
MOMENTUM      = 0.9        # SGD momentum — from AGZ paper
TRAIN_STEPS   = 200        # gradient steps per iteration
GAMES_PER_ITER= 50         # self-play games per iteration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 2. Game ───────────────────────────────────────────────────────────────────

class Gomoku:
    """
    Board: 9x9 numpy array. 0=empty, 1=black, 2=white.
    Always represents from the perspective of whoever is to move — see state().
    """
    def __init__(self):
        self.board  = np.zeros((BOARD, BOARD), dtype=np.int8)
        self.player = 1          # current player to move
        self.last   = None       # last move (r, c)
        self.n_moves= 0

    def clone(self):
        g = Gomoku.__new__(Gomoku)
        g.board   = self.board.copy()
        g.player  = self.player
        g.last    = self.last
        g.n_moves = self.n_moves
        return g

    def legal(self):
        """Flat action indices of empty squares."""
        rows, cols = np.where(self.board == 0)
        return list(rows * BOARD + cols)

    def move(self, action):
        r, c = divmod(action, BOARD)
        self.board[r, c] = self.player
        self.last    = (r, c)
        self.n_moves += 1
        self.player  = 3 - self.player   # 1↔2

    def _check_win(self, r, c):
        p = self.board[r, c]
        for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
            cnt = 1
            for s in (1, -1):
                nr, nc = r + s*dr, c + s*dc
                while 0 <= nr < BOARD and 0 <= nc < BOARD and self.board[nr,nc] == p:
                    cnt += 1; nr += s*dr; nc += s*dc
            if cnt >= WIN:
                return True
        return False

    def terminal(self):
        """Returns (is_terminal, winner).  winner: 1|2 or 0 for draw."""
        if self.last and self._check_win(*self.last):
            return True, 3 - self.player   # player who just moved (already toggled)
        if self.n_moves == BOARD * BOARD:
            return True, 0
        return False, None

    def state(self):
        """
        Input tensor [3, 9, 9] always from current player's perspective:
          plane 0 — current player's stones
          plane 1 — opponent's stones
          plane 2 — constant 1.0 if current player is black (player 1), else 0.0

        Perspective-normalised: network sees identical structure regardless of colour.
        """
        s = np.zeros((N_PLANES, BOARD, BOARD), dtype=np.float32)
        s[0] = (self.board == self.player)
        s[1] = (self.board == 3 - self.player)
        s[2] = float(self.player == 1)
        return s


# ── 3. Network ────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f)
        )
    def forward(self, x):
        return F.relu(self.net(x) + x, inplace=True)


class AZNet(nn.Module):
    """
    Dual-headed ResNet — identical in design to AGZ/AZ.
    Policy head  → logits over all BOARD² actions (illegal moves masked at inference).
    Value head   → scalar ∈ (-1, 1) estimating win probability for current player.
    """
    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, N_FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(N_FILTERS), nn.ReLU(inplace=True)
        )
        self.tower = nn.Sequential(*[ResBlock(N_FILTERS) for _ in range(N_BLOCKS)])

        # Policy head: 1×1 conv → flatten → FC
        self.p_conv = nn.Sequential(
            nn.Conv2d(N_FILTERS, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True)
        )
        self.p_fc = nn.Linear(2 * BOARD * BOARD, BOARD * BOARD)

        # Value head: 1×1 conv → FC(64) → FC(1)
        self.v_conv = nn.Sequential(
            nn.Conv2d(N_FILTERS, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )
        self.v_fc = nn.Sequential(
            nn.Linear(BOARD * BOARD, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        p = self.p_fc(self.p_conv(x).flatten(1))           # raw logits
        v = self.v_fc(self.v_conv(x).flatten(1))           # ∈ (-1,1)
        return p, v


# ── 4. MCTS ───────────────────────────────────────────────────────────────────

class Node:
    """One node per (state, action-taken-to-reach-it)."""
    __slots__ = ('P', 'N', 'W', 'children', 'expanded')
    def __init__(self, prior: float):
        self.P        = prior
        self.N        = 0
        self.W        = 0.0
        self.children : dict[int, 'Node'] = {}
        self.expanded = False

    @property
    def Q(self): return self.W / self.N if self.N else 0.0


def _net_eval(game: Gomoku, net: AZNet):
    """Single network call. Returns (masked_priors [BOARD²], value scalar)."""
    s = torch.tensor(game.state(), device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits, v = net(s)
    logits = logits[0]
    legal  = game.legal()
    mask   = torch.full((BOARD*BOARD,), float('-inf'), device=DEVICE)
    mask[legal] = 0.0
    priors = F.softmax(logits + mask, dim=0).cpu().numpy()
    return priors, v.item(), legal


def _expand(node: Node, priors, legal):
    node.expanded = True
    for a in legal:
        node.children[a] = Node(prior=float(priors[a]))


def _select(node: Node):
    """PUCT — balances Q (exploitation) with U (prior-weighted exploration)."""
    N_sqrt = node.N ** 0.5      # parent visit count sqrt (≈ from AGZ formula)
    best_score, best_a, best_child = -1e9, None, None
    for a, child in node.children.items():
        score = child.Q + C_PUCT * child.P * N_sqrt / (1 + child.N)
        if score > best_score:
            best_score, best_a, best_child = score, a, child
    return best_a, best_child


def _backup(path: list, value: float):
    """
    Walk path from leaf to root, flipping sign at each step.
    value is always from the perspective of the player to move at the leaf.
    """
    for node in reversed(path):
        node.N += 1
        node.W += value
        value   = -value


def mcts(game: Gomoku, net: AZNet, root_noise: bool = True) -> np.ndarray:
    """
    Run N_SIMS simulations from `game`.
    Returns visit-count policy π over all BOARD² actions.
    """
    root  = Node(prior=1.0)
    priors, _, legal = _net_eval(game, net)

    # Dirichlet noise at root: forces exploration of suboptimal moves.
    # Without this, self-play quickly collapses to a single opening repertoire.
    if root_noise and legal:
        noise = np.random.dirichlet([DIR_ALPHA] * len(legal))
        for i, a in enumerate(legal):
            priors[a] = (1 - DIR_EPS) * priors[a] + DIR_EPS * noise[i]

    _expand(root, priors, legal)

    for _ in range(N_SIMS):
        node   = root
        g      = game.clone()
        path   = [node]

        # ── Selection: descend until unexpanded leaf ──────────────────────────
        while node.expanded:
            a, node = _select(node)
            g.move(a)
            path.append(node)

        # ── Evaluation ───────────────────────────────────────────────────────
        done, winner = g.terminal()
        if done:
            # winner = player who just moved = 3 - g.player (player was toggled in move())
            # from g.player's (current player's) perspective: they lost → value = -1
            value = 0.0 if winner == 0 else -1.0
        else:
            priors2, value, legal2 = _net_eval(g, net)
            _expand(node, priors2, legal2)

        # ── Backup ───────────────────────────────────────────────────────────
        _backup(path, value)

    # Convert visit counts to policy vector
    pi = np.zeros(BOARD * BOARD)
    for a, child in root.children.items():
        pi[a] = child.N
    pi /= pi.sum()
    return pi


# ── 5. Self-play ──────────────────────────────────────────────────────────────

def play_game(net: AZNet) -> list[tuple]:
    """
    Play one complete game against itself.
    Returns list of (state_planes, pi, z) training examples.
    """
    net.eval()
    game    = Gomoku()
    history = []   # (state_planes, pi, player_who_moved)

    while True:
        pi     = mcts(game, net, root_noise=True)
        move_n = game.n_moves

        # Temperature schedule: sample ∝ visit counts for first TEMP_MOVES,
        # then greedy. Ensures early-game diversity in training data.
        if move_n < TEMP_MOVES:
            action = np.random.choice(BOARD * BOARD, p=pi)
        else:
            action = int(np.argmax(pi))

        history.append((game.state(), pi.copy(), game.player))
        game.move(action)

        done, winner = game.terminal()
        if done:
            examples = []
            for s, pi_h, player in history:
                if winner == 0:
                    z = 0.0
                else:
                    z = 1.0 if player == winner else -1.0
                examples.append((s, pi_h, np.float32(z)))
            return examples


# ── 6. Training ───────────────────────────────────────────────────────────────

def train_step(net: AZNet, opt, buf: deque) -> float:
    net.train()
    batch          = random.sample(buf, BATCH_SIZE)
    states, pis, zs = map(np.array, zip(*batch))

    states = torch.tensor(states, device=DEVICE)
    pis    = torch.tensor(pis,    device=DEVICE)
    zs     = torch.tensor(zs,     device=DEVICE).unsqueeze(1)

    logits, values = net(states)

    # Cross-entropy between MCTS policy π and network policy p
    # Using log_softmax + dot product (= cross-entropy for soft targets)
    policy_loss = -(pis * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    # MSE between network value estimate v and actual game outcome z
    value_loss  = F.mse_loss(values, zs)

    # Combined loss — equal weighting as in AGZ paper
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


# ── 7. Main loop ──────────────────────────────────────────────────────────────

def train(n_iters: int = 100):
    net = AZNet().to(DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    buf : deque = deque(maxlen=BUFFER_SIZE)

    print(f"AlphaZero 9x9 Gomoku | device={DEVICE} | params={sum(p.numel() for p in net.parameters()):,}")

    for it in range(1, n_iters + 1):
        # ── Self-play ─────────────────────────────────────────────────────────
        net.eval()
        for _ in range(GAMES_PER_ITER):
            buf.extend(play_game(net))

        if len(buf) < BATCH_SIZE:
            print(f"Iter {it:3d} | collecting... ({len(buf)} examples)")
            continue

        # ── Training ─────────────────────────────────────────────────────────
        total = sum(train_step(net, opt, buf) for _ in range(TRAIN_STEPS))
        print(f"Iter {it:3d} | buf={len(buf):6d} | loss={total/TRAIN_STEPS:.4f}")

        if it % 10 == 0:
            torch.save(net.state_dict(), f"az_iter{it:04d}.pt")

    return net


if __name__ == "__main__":
    train()