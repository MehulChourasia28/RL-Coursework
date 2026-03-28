import sys
import copy
import threading
import pygame
import numpy as np
import torch

from az_gomuku5 import Gomoku, AZNet, mcts, Node, BOARD, DEVICE

# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
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

# Colours
WOOD_COLOR    = (222, 184, 135)
BLACK         = (0,   0,   0)
WHITE         = (240, 240, 240)
LINE_COLOR    = (50,  50,  50)
HOVER_COLOR   = (100, 100, 100)
PANEL_COLOR   = (250, 250, 250)
BUTTON_COLOR  = (70,  130, 180)
BUTTON_HOVER  = (100, 149, 237)
STATUS_BG     = (40,  40,  40)
STATUS_TEXT   = (220, 220, 220)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def px_to_cell(mx, my):
    return round((my - OFFSET) / CELL_SIZE), round((mx - OFFSET) / CELL_SIZE)

def action_to_rc(action):
    return action // BOARD, action % BOARD

def rc_to_action(r, c):
    return r * BOARD + c

def board_at_step(history, step):
    """Reconstruct board after `step` moves from history list of (action, player)."""
    board = np.zeros((BOARD, BOARD), dtype=int)
    for i in range(step):
        action, player = history[i]
        r, c = action_to_rc(action)
        board[r, c] = player
    return board


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────
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


def draw_game_over(screen, font_big, font_small, winner, modes, review_step, total_moves):
    """Status bar: result | ← step/total → | Play Again | X"""
    bar_y = HEIGHT
    screen.fill(STATUS_BG, (0, bar_y, WIDTH, STATUS_BAR_H))

    if winner == 1:
        msg = "Black Wins!"
    elif winner == 2:
        msg = "White Wins!"
    else:
        msg = "It's a Draw!"

    mouse_pos = pygame.mouse.get_pos()
    btn_h  = max(28, STATUS_BAR_H - 20)
    margin = 12
    gap    = 8
    btn_y  = bar_y + (STATUS_BAR_H - btn_h) // 2

    # Pre-calculate all widths
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

    # Layout right-to-left
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

    # Step counter pill
    pygame.draw.rect(screen, (55, 55, 55), counter_rect, border_radius=6)
    ct = font_small.render(counter_str, True, (200, 200, 200))
    screen.blit(ct, ct.get_rect(center=counter_rect.center))

    return back_rect, fwd_rect, restart_rect, quit_rect


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def play_match(p1_type, p2_type, p1_path=None, p2_path=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + STATUS_BAR_H))
    pygame.display.set_caption(f"Gomoku {BOARD}x{BOARD}")
    clock  = pygame.time.Clock()

    font_big    = pygame.font.SysFont("Arial", max(24, min(48, HEIGHT // 10)), bold=True)
    font_small  = pygame.font.SysFont("Arial", max(16, min(24, HEIGHT // 18)))
    font_status = pygame.font.SysFont("Arial", max(14, min(20, STATUS_BAR_H // 2)))

    nets = {1: None, 2: None}
    if p1_type == 'agent':
        nets[1] = AZNet().to(DEVICE)
        nets[1].load_state_dict(torch.load(p1_path, map_location=DEVICE, weights_only=True))
        nets[1].eval()
    if p2_type == 'agent':
        nets[2] = AZNet().to(DEVICE)
        nets[2].load_state_dict(torch.load(p2_path, map_location=DEVICE, weights_only=True))
        nets[2].eval()

    modes = {1: p1_type, 2: p2_type}

    # Whether this matchup requires a human-selected opening move for Black
    agent_vs_agent = (p1_type == 'agent' and p2_type == 'agent')

    def reset():
        return (
            Gomoku(),
            {1: Node(prior=1.0), 2: Node(prior=1.0)},
            None,       # last_action
            'playing',
            0,          # end_winner
            [],         # history: list of (action, player)
            0,          # review_step (only used in 'done' state)
        )

    game, roots, last_action, state, end_winner, history, review_step = reset()

    agent_result   = [None]
    agent_thinking = [False]

    def run_agent(g, net, root):
        # Deep copy so the search thread owns its own game/root state,
        # preventing corruption if the main thread advances the game concurrently.
        pi = mcts(g.clone(), net, copy.deepcopy(root), root_noise=False, n_sims=400)
        agent_result[0] = int(np.argmax(pi))
        agent_thinking[0] = False

    running  = True
    back_btn = fwd_btn = restart_btn = quit_btn = None

    while running:
        p = game.player

        # True only on the very first move of an agent-vs-agent game
        waiting_first = (
            agent_vs_agent
            and state == 'playing'
            and len(history) == 0
        )

        # Kick off agent (suppressed while waiting for human to select opening)
        if state == 'playing' and modes[p] == 'agent' \
                and not agent_thinking[0] and agent_result[0] is None \
                and not waiting_first:
            agent_thinking[0] = True
            threading.Thread(target=run_agent,
                             args=(game, nets[p], roots[p]), daemon=True).start()

        # Apply agent move
        if state == 'playing' and modes[p] == 'agent' \
                and agent_result[0] is not None and not waiting_first:
            action = agent_result[0]
            agent_result[0] = None
            for i in (1, 2):
                if modes[i] == 'agent':
                    roots[i] = roots[i].children.get(action, Node(prior=1.0))
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
                    game, roots, last_action, state, end_winner, history, review_step = reset()
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
                        game, roots, last_action, state, end_winner, history, review_step = reset()
                        agent_result[0] = None
                        agent_thinking[0] = False
                    elif quit_btn and quit_btn.collidepoint(event.pos):
                        running = False

                elif waiting_first:
                    mx, my = event.pos
                    if my < HEIGHT:
                        row, col = px_to_cell(mx, my)
                        if 0 <= row < BOARD and 0 <= col < BOARD:
                            action = rc_to_action(row, col)
                            if action in game.legal():
                                for i in (1, 2):
                                    roots[i] = roots[i].children.get(action, Node(prior=1.0))
                                history.append((action, p))
                                last_action = action
                                game.move(action)
                                # No terminal check needed — first move can't end the game

                elif state == 'playing' and modes[p] == 'human':
                    mx, my = event.pos
                    if my < HEIGHT:
                        row, col = px_to_cell(mx, my)
                        if 0 <= row < BOARD and 0 <= col < BOARD:
                            action = rc_to_action(row, col)
                            if action in game.legal():
                                for i in (1, 2):
                                    if modes[i] == 'agent':
                                        roots[i] = roots[i].children.get(action, Node(prior=1.0))
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
                screen, font_big, font_small, end_winner, modes,
                review_step, len(history))
        else:
            draw_board(screen, game.board, last_action)

            if (modes[p] == 'human' or waiting_first) and my < HEIGHT:
                draw_hover(screen, game.board, mx, my)

            if waiting_first:
                draw_status(screen, font_status, "Select Black's opening move")
            elif modes[p] == 'human':
                who = "Black (you)" if p == 1 else "White (you)"
                draw_status(screen, font_status, f"{who} — click to place")
            else:
                dots = "." * (1 + (pygame.time.get_ticks() // 400) % 3)
                who  = "Black" if p == 1 else "White"
                draw_status(screen, font_status, f"Agent ({who}) thinking{dots}")

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Select Matchup:")
    print("1. Human (Black) vs Agent (White)")
    print("2. Agent (Black) vs Human (White)")
    print("3. Agent (Black) vs Agent (White)")
    print("4. Human (Black) vs Human (White)")

    choice = input("\nChoice (1-4): ").strip()

    if choice == '1':
        p2 = input("Path to Agent (White) model: ")
        play_match('human', 'agent', p2_path=p2)
    elif choice == '2':
        p1 = input("Path to Agent (Black) model: ")
        play_match('agent', 'human', p1_path=p1)
    elif choice == '3':
        p1 = input("Path to Agent (Black) model: ")
        p2 = input("Path to Agent (White) model: ")
        play_match('agent', 'agent', p1_path=p1, p2_path=p2)
    elif choice == '4':
        play_match('human', 'human')
    else:
        print("Invalid choice.")