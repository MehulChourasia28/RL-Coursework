# models_az5/az_iter0118.pt
# models_az5/az_iter0150.pt

import numpy as np
import torch
import torch.nn.functional as F

from az_gomuku5 import (Gomoku, AZNet, Node, _expand, _select, _backup,
                         BOARD, DEVICE)


def load_net(path):
    net = AZNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    net.eval()
    return net


# ── Batched MCTS for eval ────────────────────────────────────────────────────

def _batched_mcts(games, roots, net, n_sims):
    """Run n_sims of MCTS for multiple games sharing one net. Returns greedy actions."""
    n = len(games)

    # Expand unexpanded roots in one batch
    unexpanded = [i for i in range(n) if not roots[i].expanded]
    if unexpanded:
        states = torch.tensor(
            np.stack([games[i].state() for i in unexpanded]), device=DEVICE)
        with torch.no_grad():
            logits, _ = net(states)
        for idx, i in enumerate(unexpanded):
            legal = games[i].legal()
            mask = torch.full((BOARD * BOARD,), float('-inf'), device=DEVICE)
            mask[legal] = 0.0
            priors = F.softmax(logits[idx] + mask, dim=0).cpu().numpy()
            _expand(roots[i], priors, legal)

    # Simulations
    for _ in range(n_sims):
        paths = []
        actions_lists = []
        leaf_states = []
        dones = []
        winners = []

        # Selection
        for i in range(n):
            node = roots[i]
            path = [node]
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

        # Batched evaluation
        values = np.zeros(n)
        if leaf_states:
            states_t = torch.tensor(np.stack(leaf_states), device=DEVICE)
            with torch.no_grad():
                logits, net_vals = net(states_t)

            valid_idx = 0
            for i in range(n):
                if not dones[i]:
                    legal = games[i].legal()
                    mask = torch.full((BOARD * BOARD,), float('-inf'), device=DEVICE)
                    mask[legal] = 0.0
                    priors = F.softmax(logits[valid_idx] + mask, dim=0).cpu().numpy()
                    _expand(paths[i][-1], priors, legal)
                    values[i] = net_vals[valid_idx].item()
                    valid_idx += 1
                else:
                    values[i] = 0.0 if winners[i] == 0 else -1.0
        else:
            for i in range(n):
                if dones[i]:
                    values[i] = 0.0 if winners[i] == 0 else -1.0

        # Backup and undo
        for i in range(n):
            _backup(paths[i], values[i])
            for _ in actions_lists[i]:
                games[i].undo_move()

    # Greedy action selection
    actions = []
    for i in range(n):
        pi = np.zeros(BOARD * BOARD)
        for a, child in roots[i].children.items():
            pi[a] = child.N
        actions.append(int(np.argmax(pi)))
    return actions


# ── Main eval loop ───────────────────────────────────────────────────────────

def run_eval(path_a, path_b, n=20, sims_a=400, sims_b=400):
    print(f"Loading model A: {path_a}")
    net_a = load_net(path_a)
    print(f"Loading model B: {path_b}")
    net_b = load_net(path_b)
    print(f"Device: {DEVICE} | Games: {n*2} ({n} per side) | Sims A: {sims_a} | Sims B: {sims_b}")

    total = n * 2
    games       = [Gomoku() for _ in range(total)]
    # First n games: A=black(1), B=white(2). Second n: B=black(1), A=white(2).
    game_nets   = []
    for i in range(total):
        if i < n:
            game_nets.append({1: net_a, 2: net_b})
        else:
            game_nets.append({1: net_b, 2: net_a})

    roots       = [{1: Node(1.0), 2: Node(1.0)} for _ in range(total)]
    active      = list(range(total))
    results_arr = [None] * total   # (winner, move_count)
    move_counts = [0] * total

    # Random 3-move openings, paired: game i and game i+n share the same opening
    openings = []
    for _ in range(n):
        temp = Gomoku()
        seq = []
        for _ in range(3):
            legal = temp.legal()
            action = int(np.random.choice(legal))
            temp.move(action)
            seq.append(action)
        openings.append(seq)

    for i in range(total):
        seq = openings[i % n]
        for action in seq:
            for p in (1, 2):
                roots[i][p] = roots[i][p].children.get(action, Node(1.0))
            games[i].move(action)
        move_counts[i] = 3

    completed = 0
    move_round = 0
    print(f"\nStarting eval ({len(active)} active games)...")
    while active:
        move_round += 1
        # Group active games by which net is to move
        groups = {}
        for i in active:
            p = games[i].player
            net = game_nets[i][p]
            net_id = id(net)
            if net_id not in groups:
                groups[net_id] = (net, [])
            groups[net_id][1].append(i)

        print(f"  Round {move_round:3d} | {len(active)} active games | computing moves...", end="", flush=True)

        # Run batched MCTS for each net group
        net_sims = {id(net_a): sims_a, id(net_b): sims_b}
        for net_id, (net, group) in groups.items():
            g_list = [games[i] for i in group]
            r_list = [roots[i][games[i].player] for i in group]
            chosen = _batched_mcts(g_list, r_list, net, net_sims[net_id])

            for idx, i in enumerate(group):
                action = chosen[idx]
                for p in (1, 2):
                    roots[i][p] = roots[i][p].children.get(action, Node(1.0))
                games[i].move(action)
                move_counts[i] += 1

        finished_this_round = 0
        # Check terminals and report finished games
        new_active = []
        for i in active:
            done, winner = games[i].terminal()
            if done:
                results_arr[i] = (winner, move_counts[i])
                completed += 1
                finished_this_round += 1

                a_is_black = (i < n)
                a_won = (winner == 1) if a_is_black else (winner == 2)
                b_won = (winner == 2) if a_is_black else (winner == 1)
                status = "A wins" if a_won else ("B wins" if b_won else "draw")
                side = "A=Black  B=White" if a_is_black else "A=White  B=Black"
                print(f"  [{completed:3d}/{total}]  {side}  →  {status:<8}  {move_counts[i]} moves")
            else:
                new_active.append(i)
        if finished_this_round == 0:
            print(f" done")
        else:
            print(f" done ({finished_this_round} game(s) finished)")
        active = new_active

    # ── Summary ──────────────────────────────────────────────────────────────
    results = {
        'A': {'black': [0, 0, 0], 'white': [0, 0, 0]},
        'B': {'black': [0, 0, 0], 'white': [0, 0, 0]},
    }

    for i in range(total):
        winner, _ = results_arr[i]
        a_is_black = (i < n)

        if a_is_black:
            col_a, col_b = 'black', 'white'
            a_won = (winner == 1)
            b_won = (winner == 2)
        else:
            col_a, col_b = 'white', 'black'
            a_won = (winner == 2)
            b_won = (winner == 1)

        draw = (winner == 0)
        results['A'][col_a][0 if a_won else (1 if draw else 2)] += 1
        results['B'][col_b][0 if b_won else (1 if draw else 2)] += 1

    def fmt(wdl):
        return f"W{wdl[0]}  D{wdl[1]}  L{wdl[2]}"

    a_total_wins = results['A']['black'][0] + results['A']['white'][0]
    b_total_wins = results['B']['black'][0] + results['B']['white'][0]
    total_draws  = results['A']['black'][1] + results['A']['white'][1]

    print()
    print("=" * 46)
    print(f"  {'':4}  {'as Black':>18}  {'as White':>18}")
    print(f"  {'A':4}  {fmt(results['A']['black']):>18}  {fmt(results['A']['white']):>18}")
    print(f"  {'B':4}  {fmt(results['B']['black']):>18}  {fmt(results['B']['white']):>18}")
    print("-" * 46)
    print(f"  Total wins  →  A: {a_total_wins}   B: {b_total_wins}   Draws: {total_draws}")
    print("=" * 46)


if __name__ == "__main__":
    path_a = input("Path to model A: ").strip()
    sims_a = int(input("MCTS sims for A [400]: ").strip() or 400)
    path_b = input("Path to model B: ").strip()
    sims_b = int(input("MCTS sims for B [400]: ").strip() or 400)
    run_eval(path_a, path_b, n=50, sims_a=sims_a, sims_b=sims_b)
