"""
agent.py — Reinforcement Learning Gomoku Agent

Architecture:
 1. Pattern-based board evaluation (learned weights via TD(λ))
 2. Monte Carlo Tree Search (MCTS) for move selection
 3. Opponent modelling — tracks the human's move tendencies
 4. Persistent learning — weights saved to a JSON file

The agent gets stronger over time as it learns from each game.
"""

import json
import math
import os
import random

from Game import BOARD_SIZE, BLACK, EMPTY, WHITE, GomokuGame

BRAIN_FILE = "gomoku_brain.json"


class GomokuAgent:
    def __init__(self):
        # ── Learned weights (persist across sessions) ─────────────────
        self.pattern_weights: dict[str, float] = {}
        self.opponent_model: dict[str, float] = {}
        self.games_played = 0
        self.learning_rate = 0.08
        self.discount_factor = 0.95
        self.exploration_rate = 1.4  # UCB exploration constant for MCTS

        # ── MCTS settings (scaled by difficulty) ──────────────────────
        self.mcts_iterations = 800
        self.difficulty = 3  # 1-5 scale

        # ── Episode memory for TD learning ────────────────────────────
        self.episode_states: list[dict] = []
        self.episode_actions: list = []

        self.load_brain()

    # ═══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def set_difficulty(self, level: int):
        """Set difficulty (1-5). Adjusts MCTS iterations and evaluation depth."""
        self.difficulty = max(1, min(5, level))
        self.mcts_iterations = [100, 300, 800, 1500, 3000][self.difficulty - 1]

    def choose_move(self, game: GomokuGame) -> dict:
        """
        Choose the best move for WHITE given the current game state.
        Returns dict: { row, col, stats }.
        """
        board = game.clone_board()
        relevant_moves = game.get_relevant_moves(board)

        if not relevant_moves:
            c = BOARD_SIZE // 2
            return {"row": c, "col": c, "stats": {"iterations": 0, "depth": 0}}

        if len(relevant_moves) == 1:
            r, c = relevant_moves[0]
            return {"row": r, "col": c, "stats": {"iterations": 0, "depth": 0}}

        # ── Check for immediate wins/blocks ──────────────────────────
        tactical = self._find_tactical_move(board, relevant_moves)
        if tactical:
            return {**tactical, "stats": {"iterations": 0, "depth": 1, "tactical": True}}

        # ── MCTS search ──────────────────────────────────────────────
        result = self._mcts_search(board, relevant_moves)

        # ── Record state for learning ─────────────────────────────────
        self.episode_states.append(self._extract_features(board))

        return result

    def learn_from_game(self, reward: float, game: GomokuGame):
        """
        Called when a game ends. Performs TD(λ) weight update.
        reward: +1 for agent win, -1 for agent loss, 0 for draw
        """
        self._td_update(reward)
        self._update_opponent_model(game)
        self.games_played += 1
        self.episode_states = []
        self.episode_actions = []
        self.save_brain()

    def record_human_move(self, board: list, row: int, col: int):
        """Record the human's move for opponent modelling."""
        key = self._get_local_pattern_key(board, row, col, BLACK)
        self.opponent_model[key] = self.opponent_model.get(key, 0) + 1

    def get_stats(self) -> dict:
        return {
            "games_played": self.games_played,
            "patterns_known": len(self.pattern_weights),
        }

    def reset_brain(self):
        self.pattern_weights = {}
        self.opponent_model = {}
        self.games_played = 0
        self.episode_states = []
        self.episode_actions = []
        if os.path.exists(BRAIN_FILE):
            os.remove(BRAIN_FILE)

    # ═══════════════════════════════════════════════════════════
    #  TACTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════

    def _find_tactical_move(self, board: list, moves: list) -> dict | None:
        """Check for immediate winning moves or must-block moves."""
        # 1. Can we win immediately?
        for r, c in moves:
            board[r][c] = WHITE
            if GomokuGame.check_win_static(board, r, c, WHITE):
                board[r][c] = EMPTY
                return {"row": r, "col": c}
            board[r][c] = EMPTY

        # 2. Must we block opponent's winning move?
        for r, c in moves:
            board[r][c] = BLACK
            if GomokuGame.check_win_static(board, r, c, BLACK):
                board[r][c] = EMPTY
                return {"row": r, "col": c}
            board[r][c] = EMPTY

        # 3. Check for double threats (open four, etc.)
        threat = self._find_threat_move(board, moves, WHITE)
        if threat:
            return threat

        # 4. Block opponent's double threats
        block = self._find_threat_move(board, moves, BLACK)
        if block:
            return block

        return None

    def _find_threat_move(self, board: list, moves: list, player: int) -> dict | None:
        """Find a move that creates an unstoppable double threat."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r, c in moves:
            board[r][c] = player
            threats = 0
            for dr, dc in directions:
                line = self._get_line(board, r, c, dr, dc, player)
                if line["count"] >= 4 and line["open_ends"] >= 1:
                    threats += 1
                if line["count"] >= 3 and line["open_ends"] >= 2:
                    threats += 1
            board[r][c] = EMPTY
            if threats >= 2:
                return {"row": r, "col": c}
        return None

    def _get_line(self, board: list, r: int, c: int, dr: int, dc: int, player: int) -> dict:
        """Analyse a line through (r,c) in direction (dr,dc)."""
        count = 1
        open_ends = 0

        # Positive direction
        i = 1
        while i < 5:
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                break
            if board[nr][nc] != player:
                break
            count += 1
            i += 1
        pr, pc = r + dr * i, c + dc * i
        if 0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE and board[pr][pc] == EMPTY:
            open_ends += 1

        # Negative direction
        i = 1
        while i < 5:
            nr, nc = r - dr * i, c - dc * i
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                break
            if board[nr][nc] != player:
                break
            count += 1
            i += 1
        mr, mc = r - dr * i, c - dc * i
        if 0 <= mr < BOARD_SIZE and 0 <= mc < BOARD_SIZE and board[mr][mc] == EMPTY:
            open_ends += 1

        return {"count": count, "open_ends": open_ends}

    # ═══════════════════════════════════════════════════════════
    #  MONTE CARLO TREE SEARCH (MCTS)
    # ═══════════════════════════════════════════════════════════

    def _mcts_search(self, root_board: list, available_moves: list) -> dict:
        root = {
            "board": [row[:] for row in root_board],
            "player": WHITE,
            "visits": 0,
            "total_value": 0.0,
            "children": [],
            "move": None,
            "parent": None,
            "untried_moves": list(available_moves),
        }

        max_depth = 0

        for _ in range(self.mcts_iterations):
            # ── Selection ────────────────────────────────────────────
            node = root
            temp_board = [row[:] for row in node["board"]]
            depth = 0

            while not node["untried_moves"] and node["children"]:
                node = self._ucb_select(node)
                temp_board[node["move"][0]][node["move"][1]] = node["parent"]["player"]
                depth += 1

            # ── Expansion ────────────────────────────────────────────
            if node["untried_moves"]:
                idx = random.randrange(len(node["untried_moves"]))
                move = node["untried_moves"].pop(idx)

                next_player = BLACK if node["player"] == WHITE else WHITE
                temp_board[move[0]][move[1]] = node["player"]

                child = {
                    "board": [row[:] for row in temp_board],
                    "player": next_player,
                    "visits": 0,
                    "total_value": 0.0,
                    "children": [],
                    "move": move,
                    "parent": node,
                    "untried_moves": self._get_smart_moves(temp_board),
                }
                node["children"].append(child)
                node = child
                depth += 1

            max_depth = max(max_depth, depth)

            # ── Simulation (rollout) ──────────────────────────────────
            rollout_value = self._rollout(temp_board, node["player"])

            # ── Backpropagation ───────────────────────────────────────
            back = node
            while back is not None:
                back["visits"] += 1
                back["total_value"] += (
                    rollout_value if back["player"] == WHITE else -rollout_value
                )
                back = back["parent"]

        # ── Select best move ──────────────────────────────────────────
        if not root["children"]:
            m = random.choice(available_moves)
            return {"row": m[0], "col": m[1], "stats": {"iterations": 0, "depth": 0}}

        best_child = max(root["children"], key=lambda ch: ch["visits"])
        return {
            "row": best_child["move"][0],
            "col": best_child["move"][1],
            "stats": {
                "iterations": self.mcts_iterations,
                "depth": max_depth,
                "win_rate": (best_child["total_value"] / best_child["visits"] + 1) / 2,
                "visits": best_child["visits"],
            },
        }

    def _ucb_select(self, node: dict) -> dict:
        """Select child using UCB1 formula."""
        best_score = float("-inf")
        best_child = None
        for child in node["children"]:
            exploitation = child["total_value"] / child["visits"]
            exploration = self.exploration_rate * math.sqrt(
                math.log(node["visits"]) / child["visits"]
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _get_smart_moves(self, board: list) -> list:
        """Get smart candidate moves (near existing stones)."""
        moves = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != EMPTY:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < BOARD_SIZE
                                and 0 <= nc < BOARD_SIZE
                                and board[nr][nc] == EMPTY
                            ):
                                moves.add((nr, nc))
        return list(moves)

    def _rollout(self, board: list, current_player: int) -> float:
        """
        Perform a semi-random rollout from the given state.
        Returns a value in [-1, 1] from WHITE's perspective.
        """
        sim_board = [row[:] for row in board]
        player = current_player
        max_rollout_depth = 20

        for _ in range(max_rollout_depth):
            moves = self._get_smart_moves(sim_board)
            if not moves:
                return 0.0  # draw

            # Check for immediate wins
            for r, c in moves:
                sim_board[r][c] = player
                if GomokuGame.check_win_static(sim_board, r, c, player):
                    return 1.0 if player == WHITE else -1.0
                sim_board[r][c] = EMPTY

            # Check for must-blocks
            opp = BLACK if player == WHITE else WHITE
            chosen = None
            for r, c in moves:
                sim_board[r][c] = opp
                if GomokuGame.check_win_static(sim_board, r, c, opp):
                    chosen = (r, c)
                    sim_board[r][c] = EMPTY
                    break
                sim_board[r][c] = EMPTY

            if not chosen:
                # Evaluation-weighted random selection
                scores = [
                    max(self._evaluate_local_move(sim_board, r, c, player), 0.01)
                    for r, c in moves
                ]
                total = sum(scores)
                rand = random.random() * total
                for i, (r, c) in enumerate(moves):
                    rand -= scores[i]
                    if rand <= 0:
                        chosen = (r, c)
                        break
                if not chosen:
                    chosen = moves[-1]

            sim_board[chosen[0]][chosen[1]] = player
            player = BLACK if player == WHITE else WHITE

        return self._evaluate_board(sim_board)

    # ═══════════════════════════════════════════════════════════
    #  BOARD EVALUATION (LEARNED + HEURISTIC)
    # ═══════════════════════════════════════════════════════════

    def _evaluate_board(self, board: list) -> float:
        """Evaluate the entire board from WHITE's perspective. Returns value in [-1, 1]."""
        score = 0.0
        features = self._extract_features(board)

        for pattern, count in features.items():
            score += self.pattern_weights.get(pattern, 0) * count

        score += self._heuristic_eval(board)
        return math.tanh(score * 0.1)

    def _heuristic_eval(self, board: list) -> float:
        """Heuristic board evaluation based on line patterns."""
        score = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == EMPTY:
                    continue
                player = board[r][c]
                sign = 1 if player == WHITE else -1

                for dr, dc in directions:
                    line = self._get_line(board, r, c, dr, dc, player)
                    cnt, ends = line["count"], line["open_ends"]
                    if cnt >= 5:
                        score += sign * 10000
                    elif cnt == 4 and ends == 2:
                        score += sign * 5000
                    elif cnt == 4 and ends == 1:
                        score += sign * 500
                    elif cnt == 3 and ends == 2:
                        score += sign * 200
                    elif cnt == 3 and ends == 1:
                        score += sign * 50
                    elif cnt == 2 and ends == 2:
                        score += sign * 20
                    elif cnt == 2 and ends == 1:
                        score += sign * 5

        return score * 0.001

    def _evaluate_local_move(self, board: list, row: int, col: int, player: int) -> float:
        """Evaluate a single move locally."""
        score = 1.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        board[row][col] = player
        for dr, dc in directions:
            line = self._get_line(board, row, col, dr, dc, player)
            cnt, ends = line["count"], line["open_ends"]
            if cnt >= 5:
                score += 10000
            elif cnt == 4 and ends >= 1:
                score += 500
            elif cnt == 3 and ends == 2:
                score += 100
            elif cnt == 3 and ends == 1:
                score += 20
            elif cnt == 2 and ends == 2:
                score += 10

        # Defensive value
        opp = BLACK if player == WHITE else WHITE
        board[row][col] = opp
        for dr, dc in directions:
            line = self._get_line(board, row, col, dr, dc, opp)
            cnt, ends = line["count"], line["open_ends"]
            if cnt >= 4 and ends >= 1:
                score += 400
            elif cnt == 3 and ends == 2:
                score += 80

        board[row][col] = EMPTY

        # Learned patterns
        key = self._get_local_pattern_key(board, row, col, player)
        score += self.pattern_weights.get(key, 0) * 10

        # Opponent model
        if player == WHITE:
            opp_key = self._get_local_pattern_key(board, row, col, BLACK)
            score += self.opponent_model.get(opp_key, 0) * 2

        return max(score, 0.01)

    # ═══════════════════════════════════════════════════════════
    #  FEATURE EXTRACTION (for learning)
    # ═══════════════════════════════════════════════════════════

    def _extract_features(self, board: list) -> dict:
        """Extract pattern features from the board for learning."""
        features: dict[str, int] = {}
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr, dc in directions:
                    r2 = r + dr * 4
                    c2 = c + dc * 4
                    if 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE:
                        pattern = ""
                        for i in range(5):
                            cell = board[r + dr * i][c + dc * i]
                            if cell == EMPTY:
                                pattern += "."
                            elif cell == WHITE:
                                pattern += "W"
                            else:
                                pattern += "B"
                        if pattern != ".....":
                            features[pattern] = features.get(pattern, 0) + 1

        # Position features (centre control)
        centre = BOARD_SIZE // 2
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != EMPTY:
                    dist = abs(r - centre) + abs(c - centre)
                    key = f"pos_{board[r][c]}_d{dist}"
                    features[key] = features.get(key, 0) + 1

        return features

    def _get_local_pattern_key(self, board: list, row: int, col: int, player: int) -> str:
        """Get a local 5×5 pattern key around a position."""
        key = ""
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = row + dr, col + dc
                if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                    key += "X"  # off-board
                elif r == row and c == col:
                    key += "*"  # the move position
                elif board[r][c] == EMPTY:
                    key += "."
                elif board[r][c] == player:
                    key += "S"  # same
                else:
                    key += "O"  # opponent
        return key

    # ═══════════════════════════════════════════════════════════
    #  TEMPORAL-DIFFERENCE LEARNING — TD(λ)
    # ═══════════════════════════════════════════════════════════

    def _td_update(self, reward: float):
        """Update pattern weights based on the game outcome using TD(λ)."""
        if not self.episode_states:
            return

        lam = 0.7
        traces: dict[str, float] = {}
        next_value = reward

        for features in reversed(self.episode_states):
            current_value = self._evaluate_features(features)
            td_error = next_value * self.discount_factor - current_value

            for pattern, count in features.items():
                traces[pattern] = traces.get(pattern, 0) * lam * self.discount_factor + count
                update = self.learning_rate * td_error * traces[pattern]
                new_w = self.pattern_weights.get(pattern, 0) + update
                self.pattern_weights[pattern] = max(-10.0, min(10.0, new_w))

            next_value = current_value

        # Decay learning rate slightly
        self.learning_rate = max(0.01, self.learning_rate * 0.999)

    def _evaluate_features(self, features: dict) -> float:
        score = sum(self.pattern_weights.get(p, 0) * cnt for p, cnt in features.items())
        return math.tanh(score * 0.1)

    # ═══════════════════════════════════════════════════════════
    #  OPPONENT MODELLING
    # ═══════════════════════════════════════════════════════════

    def _update_opponent_model(self, game: GomokuGame):
        total_moves = sum(self.opponent_model.values())
        if total_moves > 10000:
            # Decay old data to keep model fresh
            to_delete = []
            for key in self.opponent_model:
                self.opponent_model[key] *= 0.9
                if self.opponent_model[key] < 0.1:
                    to_delete.append(key)
            for key in to_delete:
                del self.opponent_model[key]

    # ═══════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═══════════════════════════════════════════════════════════

    def save_brain(self):
        try:
            data = {
                "pattern_weights": self.pattern_weights,
                "opponent_model": self.opponent_model,
                "games_played": self.games_played,
                "learning_rate": self.learning_rate,
            }
            with open(BRAIN_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: could not save agent brain: {e}")

    def load_brain(self):
        try:
            if os.path.exists(BRAIN_FILE):
                with open(BRAIN_FILE) as f:
                    data = json.load(f)
                self.pattern_weights = data.get("pattern_weights", {})
                self.opponent_model = data.get("opponent_model", {})
                self.games_played = data.get("games_played", 0)
                self.learning_rate = data.get("learning_rate", 0.08)
        except Exception as e:
            print(f"Warning: could not load agent brain: {e}")