"""
Test harness for predict.py — visual Pygame UI.

Modes:
  1. Human vs Predict Agent   (human plays Black, predict.py plays White)
  2. Predict Agent vs Human   (predict.py plays Black, human plays White)
  3. Tree-Reuse Agent (Black) vs Predict Agent (White)
     Same model, same sims — the only difference is tree reuse vs stateless.
     Human picks Black's opening move, then both agents auto-play.
"""

import sys, os, copy, threading
import pygame
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_gomuku5 import Gomoku, AZNet, mcts, Node, BOARD, DEVICE
from predict import predict as az_predict

# ── Default model path ──────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(_DIR, "..", "models_az5", "az_iter0150.pt")

N_SIMS = 400

# ── Layout ──────────────────────────────────────────────────────────────────
DEFAULT_CELL_SIZE = 67
MAX_BOARD_PIXELS  = 840
MIN_CELL_SIZE     = 30
STATUS_BAR_H      = 86

fit_cell  = max(1, MAX_BOARD_PIXELS // BOARD)
CELL_SIZE = max(MIN_CELL_SIZE, min(DEFAULT_CELL_SIZE, fit_cell))
OFFSET    = CELL_SIZE // 2
WIDTH     = CELL_SIZE * BOARD
HEIGHT    = CELL_SIZE * BOARD
GRID_W    = max(1, CELL_SIZE // 25)
STONE_R   = max(5, CELL_SIZE // 2 - 5)
HOVER_RW  = max(1, CELL_SIZE // 20)

WOOD_COLOR   = (222, 184, 135)
BLACK        = (0,   0,   0)
WHITE        = (240, 240, 240)
LINE_COLOR   = (50,  50,  50)
HOVER_COLOR  = (100, 100, 100)
STATUS_BG    = (40,  40,  40)
STATUS_TEXT  = (220, 220, 220)
BUTTON_COLOR = (70,  130, 180)
BUTTON_HOVER = (100, 149, 237)


# ── Helpers ─────────────────────────────────────────────────────────────────
def px_to_cell(mx, my):
    return round((my - OFFSET) / CELL_SIZE), round((mx - OFFSET) / CELL_SIZE)

def action_to_rc(a):
    return a // BOARD, a % BOARD

def rc_to_action(r, c):
    return r * BOARD + c

def board_at_step(history, step):
    board = np.zeros((BOARD, BOARD), dtype=int)
    for i in range(step):
        action, player = history[i]
        r, c = action_to_rc(action)
        board[r, c] = player
    return board


# ── Drawing ─────────────────────────────────────────────────────────────────
def draw_board(screen, board, last_action=None):
    screen.fill(WOOD_COLOR, (0, 0, WIDTH, HEIGHT))
    for i in range(BOARD):
        pygame.draw.line(screen, LINE_COLOR,
                         (OFFSET, OFFSET + i * CELL_SIZE),
                         (WIDTH - OFFSET, OFFSET + i * CELL_SIZE), GRID_W)
        pygame.draw.line(screen, LINE_COLOR,
                         (OFFSET + i * CELL_SIZE, OFFSET),
                         (OFFSET + i * CELL_SIZE, HEIGHT - OFFSET), GRID_W)

    last_r, last_c = action_to_rc(last_action) if last_action is not None else (-1, -1)
    for r in range(BOARD):
        for c in range(BOARD):
            v = board[r, c]
            if v == 0:
                continue
            x = OFFSET + c * CELL_SIZE
            y = OFFSET + r * CELL_SIZE
            pygame.draw.circle(screen, BLACK if v == 1 else WHITE, (x, y), STONE_R)
            if v == 2:
                pygame.draw.circle(screen, LINE_COLOR, (x, y), STONE_R, 1)
            if r == last_r and c == last_c:
                pygame.draw.circle(screen, WHITE if v == 1 else BLACK,
                                   (x, y), max(3, STONE_R // 5))


def draw_hover(screen, board, mx, my):
    row, col = px_to_cell(mx, my)
    if 0 <= row < BOARD and 0 <= col < BOARD and board[row, col] == 0:
        x = OFFSET + col * CELL_SIZE
        y = OFFSET + row * CELL_SIZE
        pygame.draw.circle(screen, HOVER_COLOR, (x, y), STONE_R, HOVER_RW)


def draw_status(screen, font, text):
    screen.fill(STATUS_BG, (0, HEIGHT, WIDTH, STATUS_BAR_H))
    surf = font.render(text, True, STATUS_TEXT)
    screen.blit(surf, surf.get_rect(center=(WIDTH // 2, HEIGHT + STATUS_BAR_H // 2)))


def draw_game_over(screen, font_big, font_small, winner, labels,
                   review_step, total_moves):
    bar_y = HEIGHT
    screen.fill(STATUS_BG, (0, bar_y, WIDTH, STATUS_BAR_H))

    if winner == 1:
        msg = f"Black ({labels[1]}) Wins!"
    elif winner == 2:
        msg = f"White ({labels[2]}) Wins!"
    else:
        msg = "It's a Draw!"

    mouse_pos = pygame.mouse.get_pos()
    btn_h  = max(28, STATUS_BAR_H - 20)
    margin = 12
    gap    = 8
    btn_y  = bar_y + (STATUS_BAR_H - btn_h) // 2

    arrow_w      = btn_h
    play_again_w = font_small.size("Play Again")[0] + 24
    quit_w       = btn_h
    counter_str  = f"{review_step}/{total_moves}"
    counter_w    = font_small.size("999/999")[0] + 16

    buttons_w = margin + quit_w + gap + play_again_w + gap + arrow_w + gap + counter_w + gap + arrow_w

    txt_max_x = WIDTH - buttons_w - gap
    txt       = font_big.render(msg, True, (220, 220, 220))
    txt_rect  = txt.get_rect(midleft=(margin, bar_y + STATUS_BAR_H // 2))
    if txt_rect.right > txt_max_x:
        txt_rect.right = txt_max_x
    screen.blit(txt, txt_rect)

    quit_rect    = pygame.Rect(WIDTH - margin - quit_w,             btn_y, quit_w,       btn_h)
    restart_rect = pygame.Rect(quit_rect.left - gap - play_again_w, btn_y, play_again_w, btn_h)
    fwd_rect     = pygame.Rect(restart_rect.left - gap - arrow_w,   btn_y, arrow_w,      btn_h)
    counter_rect = pygame.Rect(fwd_rect.left - gap - counter_w,     btn_y, counter_w,    btn_h)
    back_rect    = pygame.Rect(counter_rect.left - gap - arrow_w,   btn_y, arrow_w,      btn_h)

    def btn(rect, label, base, hover_col=None, disabled=False):
        col = (60, 60, 60) if disabled else (hover_col if hover_col and rect.collidepoint(mouse_pos) else base)
        pygame.draw.rect(screen, col, rect, border_radius=7)
        s = font_small.render(label, True, (120, 120, 120) if disabled else (255, 255, 255))
        screen.blit(s, s.get_rect(center=rect.center))

    btn(back_rect,    "\u2190", BUTTON_COLOR, BUTTON_HOVER, disabled=(review_step == 0))
    btn(fwd_rect,     "\u2192", BUTTON_COLOR, BUTTON_HOVER, disabled=(review_step == total_moves))
    btn(restart_rect, "Play Again", BUTTON_COLOR, BUTTON_HOVER)
    btn(quit_rect,    "X", (180, 60, 60), (200, 80, 80))

    pygame.draw.rect(screen, (55, 55, 55), counter_rect, border_radius=6)
    ct = font_small.render(counter_str, True, (200, 200, 200))
    screen.blit(ct, ct.get_rect(center=counter_rect.center))

    return back_rect, fwd_rect, restart_rect, quit_rect


# ── Agent runners ───────────────────────────────────────────────────────────
def run_predict_agent(board, player, weights_path, result):
    """Stateless predict.py — fresh tree every call."""
    row, col = az_predict(board, player, weights_path=weights_path, n_sims=N_SIMS)
    result[0] = rc_to_action(row, col)

def run_tree_reuse_agent(game, net, root, result):
    """Standard MCTS with tree reuse."""
    pi = mcts(game.clone(), net, copy.deepcopy(root), root_noise=False, n_sims=N_SIMS)
    result[0] = int(np.argmax(pi))


# ── Load model for tree-reuse agent ────────────────────────────────────────
def load_net(weights_path):
    net = AZNet().to(DEVICE)
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        net.load_state_dict(ckpt["model"])
    else:
        net.load_state_dict(ckpt)
    net.eval()
    return net


# ── Main loop ──────────────────────────────────────────────────────────────
def play(mode, weights_path):
    """
    mode: 'human_vs_predict'   — human (Black) vs predict agent (White)
          'predict_vs_human'   — predict agent (Black) vs human (White)
          'reuse_vs_predict'   — tree-reuse (Black) vs predict (White)
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + STATUS_BAR_H))
    clock  = pygame.time.Clock()

    font_big    = pygame.font.SysFont("Arial", max(24, min(48, HEIGHT // 10)), bold=True)
    font_small  = pygame.font.SysFont("Arial", max(16, min(24, HEIGHT // 18)))
    font_status = pygame.font.SysFont("Arial", max(14, min(20, STATUS_BAR_H // 2)))

    # Determine roles
    if mode == 'human_vs_predict':
        roles  = {1: 'human', 2: 'predict'}
        labels = {1: "Human", 2: "Predict"}
        caption = "Human (B) vs Predict (W)"
    elif mode == 'predict_vs_human':
        roles  = {1: 'predict', 2: 'human'}
        labels = {1: "Predict", 2: "Human"}
        caption = "Predict (B) vs Human (W)"
    else:  # reuse_vs_predict
        roles  = {1: 'reuse', 2: 'predict'}
        labels = {1: "Tree-Reuse", 2: "Predict"}
        caption = "Tree-Reuse (B) vs Predict (W)"

    pygame.display.set_caption(f"{caption} — Gomoku {BOARD}x{BOARD}")

    # Load net for tree-reuse agent (if needed)
    reuse_net = load_net(weights_path) if 'reuse' in roles.values() else None

    agent_vs_agent = ('human' not in roles.values())

    def reset():
        return (
            Gomoku(),
            Node(prior=1.0),    # tree-reuse root (only used if reuse agent active)
            None,               # last_action
            'playing',
            0,                  # end_winner
            [],                 # history
            0,                  # review_step
        )

    game, reuse_root, last_action, state, end_winner, history, review_step = reset()

    agent_result   = [None]
    agent_thinking = [False]

    running  = True
    back_btn = fwd_btn = restart_btn = quit_btn = None

    while running:
        p = game.player
        role = roles[p]

        waiting_first = (agent_vs_agent and state == 'playing' and len(history) == 0)

        # Kick off agent
        if state == 'playing' and role != 'human' \
                and not agent_thinking[0] and agent_result[0] is None \
                and not waiting_first:
            agent_thinking[0] = True
            if role == 'predict':
                threading.Thread(
                    target=run_predict_agent,
                    args=(game.board.copy(), p, weights_path, agent_result),
                    daemon=True).start()
            else:  # reuse
                threading.Thread(
                    target=run_tree_reuse_agent,
                    args=(game, reuse_net, reuse_root, agent_result),
                    daemon=True).start()

        # Apply agent move
        if state == 'playing' and role != 'human' \
                and agent_result[0] is not None and not waiting_first:
            action = agent_result[0]
            agent_result[0]   = None
            agent_thinking[0] = False

            # Advance tree-reuse root
            if reuse_net is not None:
                reuse_root = reuse_root.children.get(action, Node(prior=1.0))

            history.append((action, p))
            last_action = action
            game.move(action)
            done, winner = game.terminal()
            if done:
                state, end_winner = 'done', winner
                review_step = len(history)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r and state == 'done':
                    game, reuse_root, last_action, state, end_winner, history, review_step = reset()
                    agent_result[0] = None
                    agent_thinking[0] = False
                elif state == 'done':
                    if event.key == pygame.K_LEFT and review_step > 0:
                        review_step -= 1
                    elif event.key == pygame.K_RIGHT and review_step < len(history):
                        review_step += 1

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state == 'done':
                    if back_btn and back_btn.collidepoint(event.pos) and review_step > 0:
                        review_step -= 1
                    elif fwd_btn and fwd_btn.collidepoint(event.pos) and review_step < len(history):
                        review_step += 1
                    elif restart_btn and restart_btn.collidepoint(event.pos):
                        game, reuse_root, last_action, state, end_winner, history, review_step = reset()
                        agent_result[0] = None
                        agent_thinking[0] = False
                    elif quit_btn and quit_btn.collidepoint(event.pos):
                        running = False

                elif waiting_first or (state == 'playing' and role == 'human'):
                    mx, my = event.pos
                    if my < HEIGHT:
                        row, col = px_to_cell(mx, my)
                        if 0 <= row < BOARD and 0 <= col < BOARD:
                            action = rc_to_action(row, col)
                            if action in game.legal():
                                if reuse_net is not None:
                                    reuse_root = reuse_root.children.get(action, Node(prior=1.0))
                                history.append((action, p))
                                last_action = action
                                game.move(action)
                                done, winner = game.terminal()
                                if done:
                                    state, end_winner = 'done', winner
                                    review_step = len(history)

        # Render
        mx, my = pygame.mouse.get_pos()

        if state == 'done':
            rev_board = board_at_step(history, review_step)
            rev_last  = history[review_step - 1][0] if review_step > 0 else None
            draw_board(screen, rev_board, rev_last)
            back_btn, fwd_btn, restart_btn, quit_btn = draw_game_over(
                screen, font_big, font_small, end_winner, labels,
                review_step, len(history))
        else:
            draw_board(screen, game.board, last_action)

            if (role == 'human' or waiting_first) and my < HEIGHT:
                draw_hover(screen, game.board, mx, my)

            if waiting_first:
                draw_status(screen, font_status, "Select Black's opening move")
            elif role == 'human':
                color = "Black" if p == 1 else "White"
                draw_status(screen, font_status, f"Your turn ({color}) — click to place")
            else:
                color = "Black" if p == 1 else "White"
                dots  = "." * (1 + (pygame.time.get_ticks() // 400) % 3)
                draw_status(screen, font_status,
                            f"{labels[p]} ({color}) thinking{dots}")

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Predict.py Test Harness")
    print("-" * 30)
    print("1. Human (Black) vs Predict Agent (White)")
    print("2. Predict Agent (Black) vs Human (White)")
    print("3. Tree-Reuse Agent (Black) vs Predict Agent (White)  [same model]")

    choice = input("\nChoice (1-3): ").strip()
    path   = input(f"Model path [{DEFAULT_WEIGHTS}]: ").strip() or DEFAULT_WEIGHTS

    if choice == '1':
        play('human_vs_predict', path)
    elif choice == '2':
        play('predict_vs_human', path)
    elif choice == '3':
        play('reuse_vs_predict', path)
    else:
        print("Invalid choice.")
