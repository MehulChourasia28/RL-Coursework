"""
app.py — Terminal controller. Connects the game logic and AI agent
         to a text-based board display in the terminal.

Usage:
    python app.py [--difficulty 1-5]
"""

import argparse
import sys
import time

from Agent import GomokuAgent
from Game import BLACK, BOARD_SIZE, EMPTY, WHITE, GomokuGame


try:
    import colorama
    colorama.init()
    BLACK_STONE  = "\033[1;30m●\033[0m"   # bold dark circle
    WHITE_STONE  = "\033[1;37m○\033[0m"   # bold light circle
    WIN_BLACK    = "\033[1;31m●\033[0m"   # red highlight for winning stone
    WIN_WHITE    = "\033[1;31m○\033[0m"
    BOLD         = "\033[1m"
    RESET        = "\033[0m"
    GREEN        = "\033[32m"
    YELLOW       = "\033[33m"
    CYAN         = "\033[36m"
    RED          = "\033[31m"
    _COLOR       = True
except ImportError:
    BLACK_STONE = "X"
    WHITE_STONE = "O"
    WIN_BLACK = WIN_WHITE = "*"
    BOLD = RESET = GREEN = YELLOW = CYAN = RED = ""
    _COLOR = False

# ───────────────────────────────────────────────────────────────
#  Drawing helpers
# ───────────────────────────────────────────────────────────────

def draw_board(game: GomokuGame):
    winning_set = set(map(tuple, game.winning_cells))
    col_labels = "  " + "  ".join(chr(65 + i) for i in range(BOARD_SIZE))
    print(f"\n{BOLD}{col_labels}{RESET}")

    for r in range(BOARD_SIZE):
        row_num = str(BOARD_SIZE - r).rjust(2)
        cells = []
        for c in range(BOARD_SIZE):
            val = game.board[r][c]
            is_winning = (r, c) in winning_set
            if val == BLACK:
                cells.append(WIN_BLACK if is_winning else BLACK_STONE)
            elif val == WHITE:
                cells.append(WIN_WHITE if is_winning else WHITE_STONE)
            else:
                # Star points on 9×9
                if (r, c) in {(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)}:
                    cells.append("+")
                else:
                    cells.append("·")
        print(f"{BOLD}{row_num}{RESET} " + "  ".join(cells))

    print()


def print_status(game: GomokuGame, is_agent_thinking: bool, scores: dict):
    print(f"{CYAN}{'─'*30}{RESET}")
    print(f"  You {GREEN}{scores['player']}{RESET}  |  "
          f"Agent {RED}{scores['agent']}{RESET}  |  "
          f"Draws {YELLOW}{scores['draws']}{RESET}")
    if game.game_over:
        if game.winner == BLACK:
            print(f"  {GREEN}{BOLD}You win! 🎉{RESET}")
        elif game.winner == WHITE:
            print(f"  {RED}{BOLD}Agent wins! 🤖{RESET}")
        else:
            print(f"  {YELLOW}{BOLD}Draw! 🤝{RESET}")
    elif is_agent_thinking:
        print(f"  {YELLOW}Agent is thinking…{RESET}")
    else:
        print(f"  Your turn — enter a move (e.g. E5) or a command")
    print(f"{CYAN}{'─'*30}{RESET}")


def print_help():
    print(f"""
{BOLD}Commands:{RESET}
  <move>  — e.g. E5, A1, I9  (column letter + row number)
  undo    — take back your last move
  new     — start a new game
  diff N  — set difficulty 1-5 (current game)
  stats   — show agent learning stats
  reset   — reset agent brain and scores
  quit    — exit
""")


# ───────────────────────────────────────────────────────────────
#  Input parsing
# ───────────────────────────────────────────────────────────────

def parse_move(token: str, board_size: int = BOARD_SIZE):
    """Parse 'E5' style input → (row, col) or None."""
    token = token.strip().upper()
    if len(token) < 2:
        return None
    col_char = token[0]
    if not col_char.isalpha():
        return None
    col = ord(col_char) - 65
    try:
        row_num = int(token[1:])
    except ValueError:
        return None
    row = board_size - row_num
    if 0 <= row < board_size and 0 <= col < board_size:
        return row, col
    return None


# ───────────────────────────────────────────────────────────────
#  Main game loop
# ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gomoku — Human vs AI")
    parser.add_argument("--difficulty", type=int, default=3, choices=range(1, 6),
                        metavar="1-5", help="AI difficulty (default: 3)")
    args = parser.parse_args()

    game = GomokuGame()
    agent = GomokuAgent()
    agent.set_difficulty(args.difficulty)

    scores = {"player": 0, "agent": 0, "draws": 0}

    print(f"\n{BOLD}Gomoku — 9×9 board — first to 5 in a row wins{RESET}")
    print(f"  You play {BLACK_STONE} (Black).  AI plays {WHITE_STONE} (White).")
    print_help()

    while True:
        draw_board(game)
        print_status(game, False, scores)

        if game.game_over:
            # rewards and losses
            if game.winner == BLACK:
                reward = -1.0
                scores["player"] += 1
            elif game.winner == WHITE:
                reward = 1.0
                scores["agent"] += 1
            else:
                reward = 0.0
                scores["draws"] += 1
            agent.learn_from_game(reward, game)

            cmd = input("  New game? [Y/n] ").strip().lower()
            if cmd in ("", "y", "yes"):
                game.reset()
                continue
            else:
                print("Goodbye!")
                sys.exit(0)

        # ── Human turn ────────────────────────────────────────────
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not raw:
            continue

        token = raw.lower()

        # ── Commands ──────────────────────────────────────────────
        if token in ("quit", "exit", "q"):
            print("Goodbye!")
            sys.exit(0)

        if token in ("help", "h", "?"):
            print_help()
            continue

        if token in ("new", "restart"):
            game.reset()
            print("  New game started.")
            continue

        if token == "undo":
            if game.undo_last_turn():
                print("  Move undone.")
            else:
                print("  Nothing to undo.")
            continue

        if token.startswith("diff"):
            parts = raw.split()
            if len(parts) == 2 and parts[1].isdigit():
                level = int(parts[1])
                agent.set_difficulty(level)
                print(f"  Difficulty set to {level} "
                      f"({agent.mcts_iterations} MCTS iterations).")
            else:
                print("  Usage: diff <1-5>")
            continue

        if token == "stats":
            st = agent.get_stats()
            print(f"  Games learned: {st['games_played']}  |  "
                  f"Patterns known: {st['patterns_known']}  |  "
                  f"Learning rate: {agent.learning_rate:.4f}")
            continue

        if token == "reset":
            confirm = input("  Reset agent brain and scores? [y/N] ").strip().lower()
            if confirm in ("y", "yes"):
                agent.reset_brain()
                scores = {"player": 0, "agent": 0, "draws": 0}
                game.reset()
                print("  Brain and scores reset.")
            continue

        # ── Move input ────────────────────────────────────────────
        pos = parse_move(raw)
        if pos is None:
            print(f"  Unknown command or invalid move: '{raw}'. Type 'help' for commands.")
            continue

        row, col = pos
        if not game.is_valid_move(row, col):
            print(f"  {GomokuGame.to_notation(row, col)} is already occupied or out of bounds.")
            continue

        # Record for opponent modelling and feature learning
        agent.record_human_move(game.board, row, col)
        agent.episode_states.append(agent._extract_features(game.board))

        game.make_move(row, col)

        if game.game_over:
            draw_board(game)
            print_status(game, False, scores)
            continue

        # ── Agent turn ────────────────────────────────────────────
        print_status(game, True, scores)
        t0 = time.time()
        result = agent.choose_move(game)
        elapsed = time.time() - t0

        game.make_move(result["row"], result["col"])
        notation = GomokuGame.to_notation(result["row"], result["col"])

        stats = result.get("stats", {})
        if stats.get("tactical"):
            detail = "tactical"
        elif stats.get("depth"):
            detail = f"{stats['depth']} plies, {elapsed:.2f}s"
        else:
            detail = f"{elapsed:.2f}s"
        print(f"  Agent plays {BOLD}{notation}{RESET}  ({detail})")


if __name__ == "__main__":
    main()