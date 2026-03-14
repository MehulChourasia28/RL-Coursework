"""
game.py — Core Gomoku game logic for a 9×9 board.

Manages board state, move validation, win detection, and undo.
No UI or AI logic here — pure game rules.
"""

BOARD_SIZE = 9
EMPTY = 0
BLACK = 1   # Human player
WHITE = 2   # AI agent
WIN_LENGTH = 5


class GomokuGame:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the board to an empty state."""
        # board[row][col] — 0 = empty, 1 = black, 2 = white
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = BLACK  # Black always goes first
        self.move_history = []       # List of dicts: { row, col, player }
        self.game_over = False
        self.winner = None           # None, BLACK, WHITE, or 'draw'
        self.winning_cells = []      # List of [row, col] for the winning line

    def clone_board(self):
        """Return a deep copy of the board."""
        return [row[:] for row in self.board]

    def get_empty_positions(self, board=None):
        """Get all empty positions as (row, col) pairs."""
        if board is None:
            board = self.board
        return [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board[r][c] == EMPTY
        ]

    def is_valid_move(self, row, col, board=None):
        """Check if a position is on the board and empty."""
        if board is None:
            board = self.board
        return (
            0 <= row < BOARD_SIZE
            and 0 <= col < BOARD_SIZE
            and board[row][col] == EMPTY
        )

    def make_move(self, row, col):
        """
        Make a move. Returns True if successful.
        Automatically checks for win/draw after the move.
        """
        if self.game_over:
            return False
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player
        self.move_history.append({"row": row, "col": col, "player": self.current_player})

        # Check for win
        win_result = self.check_win(row, col, self.current_player)
        if win_result:
            self.game_over = True
            self.winner = self.current_player
            self.winning_cells = win_result
            return True

        # Check for draw (board full)
        if not self.get_empty_positions():
            self.game_over = True
            self.winner = "draw"
            return True

        # Switch player
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        return True

    def undo_last_turn(self):
        """Undo the last move (or last two moves to undo a full turn)."""
        if not self.move_history:
            return False

        # If game is over, just undo the last move
        moves_to_undo = 1 if self.game_over else 2
        self.game_over = False
        self.winner = None
        self.winning_cells = []

        for _ in range(moves_to_undo):
            if not self.move_history:
                break
            last = self.move_history.pop()
            self.board[last["row"]][last["col"]] = EMPTY

        # Determine whose turn it is
        if not self.move_history:
            self.current_player = BLACK
        else:
            last_move = self.move_history[-1]
            self.current_player = WHITE if last_move["player"] == BLACK else BLACK

        return True

    def check_win(self, row, col, player, board=None):
        """
        Check if placing `player` at (row, col) creates five in a row.
        Returns list of winning cell positions, or None.
        """
        if board is None:
            board = self.board

        directions = [
            (0, 1),   # horizontal →
            (1, 0),   # vertical ↓
            (1, 1),   # diagonal ↘
            (1, -1),  # diagonal ↙
        ]

        for dr, dc in directions:
            cells = [[row, col]]

            # Count in positive direction
            for i in range(1, WIN_LENGTH):
                r, c = row + dr * i, col + dc * i
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    break
                if board[r][c] != player:
                    break
                cells.append([r, c])

            # Count in negative direction
            for i in range(1, WIN_LENGTH):
                r, c = row - dr * i, col - dc * i
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    break
                if board[r][c] != player:
                    break
                cells.append([r, c])

            if len(cells) >= WIN_LENGTH:
                return cells

        return None

    @staticmethod
    def check_win_static(board, row, col, player):
        """Static win check on an arbitrary board state (used by AI)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, WIN_LENGTH):
                r, c = row + dr * i, col + dc * i
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) or board[r][c] != player:
                    break
                count += 1
            for i in range(1, WIN_LENGTH):
                r, c = row - dr * i, col - dc * i
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) or board[r][c] != player:
                    break
                count += 1
            if count >= WIN_LENGTH:
                return True
        return False

    def get_relevant_moves(self, board=None, range_=2):
        """
        Get interesting/relevant moves (near existing stones) for smarter search.
        Returns positions within `range_` cells of any existing stone.
        """
        if board is None:
            board = self.board

        has_stones = bool(self.move_history) or any(
            cell != EMPTY for row in board for cell in row
        )

        if not has_stones:
            center = BOARD_SIZE // 2
            return [(center, center)]

        relevant = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != EMPTY:
                    for dr in range(-range_, range_ + 1):
                        for dc in range(-range_, range_ + 1):
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < BOARD_SIZE
                                and 0 <= nc < BOARD_SIZE
                                and board[nr][nc] == EMPTY
                            ):
                                relevant.add((nr, nc))

        return list(relevant)

    @staticmethod
    def to_notation(row, col):
        """Convert row, col to algebraic notation like 'E5'."""
        col_letter = chr(65 + col)        # A-I
        row_num = BOARD_SIZE - row        # 9-1
        return f"{col_letter}{row_num}"