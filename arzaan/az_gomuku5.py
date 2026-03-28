"""
AlphaZero — 9x9 Gomoku, v5.

Base: az_gomuku2 (proven stable design, batched self-play).
Changes:
  - Network: 10 res blocks, 128 filters (~2M params vs 390k)
  - Buffer: 200k capacity (was 50k) — retains more positional diversity
  - MCTS: 400 sims (was 200) — sharper policy targets
  - LR: cosine annealing (was multi-step) — smoother decay, no wasted plateau
  - Iterations: 300 (was 150) — bigger network needs more training
  - Value head FC: 256 hidden (was 64) — proportional to larger tower
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

# ── 1. Config ─────────────────────────────────────────────────────────────────

BOARD         = 9
WIN           = 5
N_PLANES      = 3          # current stones | opponent stones | turn indicator
N_BLOCKS      = 10         # residual blocks (was 5)
N_FILTERS     = 128        # channel width (was 64)

N_SIMS        = 400        # MCTS simulations per move (was 200)
C_PUCT        = 1.5        # exploration constant
DIR_ALPHA     = 0.3        # Dirichlet α
DIR_EPS       = 0.25       # noise weight at root
TEMP_MOVES    = 10         # play stochastically for first N moves, then greedy

BUFFER_SIZE   = 200_000    # replay buffer capacity (was 50k)
BATCH_SIZE    = 256        # mini-batch size
LR            = 1e-2       # SGD lr
WEIGHT_DECAY  = 1e-4       # L2 regularisation weight
MOMENTUM      = 0.9        # SGD momentum
TRAIN_STEPS   = 200        # gradient steps per iteration
GAMES_PER_ITER= 50         # self-play games per iteration

CHECKPOINT_PATH = "models_az5/checkpoint.pt"
BUFFER_PATH     = "models_az5/buffer.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 2. Game ───────────────────────────────────────────────────────────────────

class Gomoku:
    """
    Board: 9x9 numpy array. 0=empty, 1=black, 2=white.
    Always represents from the perspective of whoever is to move.
    last_moves history and undo_move() are required by batched MCTS.
    """
    def __init__(self):
        self.board      = np.zeros((BOARD, BOARD), dtype=np.int8)
        self.player     = 1
        self.last_moves = []
        self.n_moves    = 0

    @property
    def last(self):
        return divmod(self.last_moves[-1][0], BOARD) if self.last_moves else None

    def clone(self):
        g = Gomoku.__new__(Gomoku)
        g.board      = self.board.copy()
        g.player     = self.player
        g.last_moves = self.last_moves.copy()
        g.n_moves    = self.n_moves
        return g

    def legal(self):
        rows, cols = np.where(self.board == 0)
        return list(rows * BOARD + cols)

    def move(self, action):
        r, c = divmod(action, BOARD)
        self.board[r, c] = self.player
        self.last_moves.append((action, self.player))
        self.n_moves += 1
        self.player = 3 - self.player

    def undo_move(self):
        if not self.last_moves: return
        action, p = self.last_moves.pop()
        r, c = divmod(action, BOARD)
        self.board[r, c] = 0
        self.n_moves -= 1
        self.player = p

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
        if self.last and self._check_win(*self.last):
            return True, 3 - self.player
        if self.n_moves == BOARD * BOARD:
            return True, 0
        return False, None

    def state(self):
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
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, N_FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(N_FILTERS), nn.ReLU(inplace=True)
        )
        self.tower = nn.Sequential(*[ResBlock(N_FILTERS) for _ in range(N_BLOCKS)])

        self.p_conv = nn.Sequential(
            nn.Conv2d(N_FILTERS, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True)
        )
        self.p_fc = nn.Linear(2 * BOARD * BOARD, BOARD * BOARD)

        self.v_conv = nn.Sequential(
            nn.Conv2d(N_FILTERS, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )
        self.v_fc = nn.Sequential(
            nn.Linear(BOARD * BOARD, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        p = self.p_fc(self.p_conv(x).flatten(1))
        v = self.v_fc(self.v_conv(x).flatten(1))
        return p, v


# ── 4. MCTS ───────────────────────────────────────────────────────────────────

class Node:
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
    N_sqrt = node.N ** 0.5
    best_score, best_a, best_child = -1e9, None, None
    for a, child in node.children.items():
        score = child.Q + C_PUCT * child.P * N_sqrt / (1 + child.N)
        if score > best_score:
            best_score, best_a, best_child = score, a, child
    return best_a, best_child


def _backup(path: list, value: float):
    for node in reversed(path):
        value = -value
        node.N += 1
        node.W += value


def mcts(game: Gomoku, net: AZNet, root: Node, root_noise: bool = True, n_sims: int = N_SIMS) -> np.ndarray:
    """Standard MCTS using clone() — used by the pygame UI, not during training."""
    if not root.expanded:
        priors, _, legal = _net_eval(game, net)
        _expand(root, priors, legal)

    original_priors = {}
    if root_noise and root.children:
        legal = list(root.children.keys())
        noise = np.random.dirichlet([DIR_ALPHA] * len(legal))
        for i, a in enumerate(legal):
            original_priors[a] = root.children[a].P
            root.children[a].P = (1 - DIR_EPS) * original_priors[a] + DIR_EPS * noise[i]

    for _ in range(n_sims):
        node = root
        g    = game.clone()
        path = [node]
        while node.expanded:
            a, node = _select(node)
            g.move(a)
            path.append(node)
        done, winner = g.terminal()
        if done:
            value = 0.0 if winner == 0 else -1.0
        else:
            priors2, value, legal2 = _net_eval(g, net)
            _expand(node, priors2, legal2)
        _backup(path, value)

    for a, p in original_priors.items():
        root.children[a].P = p

    pi = np.zeros(BOARD * BOARD)
    for a, child in root.children.items():
        pi[a] = child.N
    pi /= pi.sum()
    return pi


# ── 5. Self-play (batched) ────────────────────────────────────────────────────

def get_symmetries(state_planes: np.ndarray, pi: np.ndarray) -> list[tuple]:
    """Generates 8x data through dihedral reflections and rotations."""
    pi_board = pi.reshape(BOARD, BOARD)
    syms = []
    for i in range(4):
        for flip in [False, True]:
            s = np.rot90(state_planes, k=i, axes=(1, 2))
            p = np.rot90(pi_board, k=i)
            if flip:
                s = np.flip(s, axis=2)
                p = np.flip(p, axis=1)
            syms.append((s.copy(), p.flatten().copy()))
    return syms


def play_games_batched(net: AZNet, n_games: int, n_sims: int) -> list[tuple]:
    """Plays n_games simultaneously using batched MCTS inference and fast rollbacks."""
    net.eval()
    games     = [Gomoku() for _ in range(n_games)]
    roots     = [Node(prior=1.0) for _ in range(n_games)]
    histories = [[] for _ in range(n_games)]
    active    = list(range(n_games))
    examples  = []

    while active:
        # 1. Evaluate unexpanded roots in a single batch
        unexpanded = [i for i in active if not roots[i].expanded]
        if unexpanded:
            states = torch.tensor(np.stack([games[i].state() for i in unexpanded]), device=DEVICE)
            with torch.no_grad():
                logits, _ = net(states)
            for idx, i in enumerate(unexpanded):
                legal = games[i].legal()
                mask  = torch.full((BOARD*BOARD,), float('-inf'), device=DEVICE)
                mask[legal] = 0.0
                priors = F.softmax(logits[idx] + mask, dim=0).cpu().numpy()
                _expand(roots[i], priors, legal)

        # 2. Add root noise
        original_priors = {}
        for i in active:
            if roots[i].children:
                legal = list(roots[i].children.keys())
                noise = np.random.dirichlet([DIR_ALPHA] * len(legal))
                original_priors[i] = {}
                for idx, a in enumerate(legal):
                    original_priors[i][a] = roots[i].children[a].P
                    roots[i].children[a].P = (1 - DIR_EPS) * original_priors[i][a] + DIR_EPS * noise[idx]

        # 3. MCTS simulations
        for _ in range(n_sims):
            paths         = []
            actions_lists = []
            leaf_states   = []
            dones         = []
            winners       = []

            # Step A: Selection (CPU traversal with fast rollbacks)
            for i in active:
                node    = roots[i]
                path    = [node]
                actions = []
                while node.expanded:
                    a, node = _select(node)
                    games[i].move(a)
                    actions.append(a)
                    path.append(node)
                paths.append(path)
                actions_lists.append(actions)
                done, winner = games[i].terminal()
                dones.append(done)
                winners.append(winner)
                if not done:
                    leaf_states.append(games[i].state())

            # Step B: Batched evaluation (single GPU forward pass)
            values = np.zeros(len(active))
            if leaf_states:
                states_tensor = torch.tensor(np.stack(leaf_states), device=DEVICE)
                with torch.no_grad():
                    logits, net_vals = net(states_tensor)

            valid_idx = 0
            for local_idx, i in enumerate(active):
                if not dones[local_idx]:
                    legal = games[i].legal()
                    mask  = torch.full((BOARD*BOARD,), float('-inf'), device=DEVICE)
                    mask[legal] = 0.0
                    priors = F.softmax(logits[valid_idx] + mask, dim=0).cpu().numpy()
                    _expand(paths[local_idx][-1], priors, legal)
                    values[local_idx] = net_vals[valid_idx].item()
                    valid_idx += 1
                else:
                    values[local_idx] = 0.0 if winners[local_idx] == 0 else -1.0

            # Step C: Backup and undo
            for local_idx, i in enumerate(active):
                _backup(paths[local_idx], values[local_idx])
                for _ in actions_lists[local_idx]:
                    games[i].undo_move()

        # Restore root priors
        for i in active:
            if i in original_priors:
                for a, p in original_priors[i].items():
                    roots[i].children[a].P = p

        # 4. Advance games
        new_active = []
        for i in active:
            pi = np.zeros(BOARD * BOARD)
            for a, child in roots[i].children.items():
                pi[a] = child.N
            pi /= pi.sum()
            # Absorb float-precision errors to prevent np.random.choice crashes
            max_idx = np.argmax(pi)
            pi[max_idx] += 1.0 - np.sum(pi)

            move_n = games[i].n_moves
            action = np.random.choice(BOARD * BOARD, p=pi) if move_n < TEMP_MOVES else int(np.argmax(pi))

            histories[i].append((games[i].state(), pi.copy(), games[i].player))
            roots[i] = roots[i].children.get(action, Node(prior=1.0))
            games[i].move(action)

            done, winner = games[i].terminal()
            if done:
                for s, pi_h, player in histories[i]:
                    z = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
                    for sym_s, sym_pi in get_symmetries(s, pi_h):
                        examples.append((sym_s, sym_pi, np.float32(z)))
            else:
                new_active.append(i)

        active = new_active

    return examples


# ── 6. Training ───────────────────────────────────────────────────────────────

def train_step(net: AZNet, opt, buf: deque) -> float:
    net.train()
    batch  = random.sample(buf, BATCH_SIZE)
    states, pis, zs = map(np.array, zip(*batch))

    states = torch.tensor(states, device=DEVICE)
    pis    = torch.tensor(pis,    device=DEVICE)
    zs     = torch.tensor(zs,     device=DEVICE).unsqueeze(1)

    logits, values = net(states)

    policy_loss = -(pis * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss  = F.mse_loss(values, zs)
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), policy_loss.item(), value_loss.item()


# ── 7. Main loop ──────────────────────────────────────────────────────────────

def train(n_iters: int = 300):
    os.makedirs("models_az5", exist_ok=True)

    net = AZNet().to(DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=5e-4)
    buf: deque = deque(maxlen=BUFFER_SIZE)

    start_iter = 1

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["iter"] + 1
        print(f"Resumed from iteration {ckpt['iter']}")

    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, "rb") as f:
            buf = pickle.load(f)
        print(f"Loaded replay buffer ({len(buf)} examples)")

    print(f"AlphaZero 9x9 Gomoku v5 | device={DEVICE} | params={sum(p.numel() for p in net.parameters()):,}")

    for it in range(start_iter, n_iters + 1):
        net.eval()
        buf.extend(play_games_batched(net, n_games=GAMES_PER_ITER, n_sims=N_SIMS))

        if len(buf) < BATCH_SIZE:
            print(f"Iter {it:3d} | collecting... ({len(buf)} examples)")
            continue

        tot_loss = tot_pol = tot_val = 0.0
        for _ in range(TRAIN_STEPS):
            l, p, v = train_step(net, opt, buf)
            tot_loss += l; tot_pol += p; tot_val += v
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Iter {it:3d} | buf={len(buf):6d} | loss={tot_loss/TRAIN_STEPS:.4f} "
              f"(pol={tot_pol/TRAIN_STEPS:.4f} val={tot_val/TRAIN_STEPS:.4f}) | lr={current_lr:.6f}")

        torch.save(net.state_dict(), f"models_az5/az_iter{it:04d}.pt")

        torch.save({
            "iter":      it,
            "model":     net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, CHECKPOINT_PATH)

        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(buf, f)

    return net


if __name__ == "__main__":
    train()
