import sys
import time
from Game import GomokuGame, BLACK, WHITE, BOARD_SIZE
from Agent import GomokuAgent

NUM_GAMES = 10
MCTS_ITERATIONS = 100  # higher means agent "thinks" more

agent = GomokuAgent()
agent.mcts_iterations = MCTS_ITERATIONS
print(f"Starting self-play: {NUM_GAMES} games  |  "
      f"{agent.mcts_iterations} MCTS iterations/move", flush=True)

for i in range(NUM_GAMES):
    game = GomokuGame()
    move_count = 0
    game_start = time.time()
    print(f"\n── Game {i+1}/{NUM_GAMES} ──────────────────────", flush=True)

    while not game.game_over:
        move_count += 1
        result = agent.choose_move(game)
        row, col = result["row"], result["col"]
        notation = GomokuGame.to_notation(row, col)
        player_name = "BLACK" if game.current_player == BLACK else "WHITE"
        print(f"  Move {move_count:>2}  {player_name} → {notation}", flush=True)
        game.make_move(row, col)

    elapsed = time.time() - game_start
    if game.winner == WHITE:
        outcome = "WHITE wins"
        reward = 1.0
    elif game.winner == BLACK:
        outcome = "BLACK wins"
        reward = -1.0
    else:
        outcome = "Draw"
        reward = 0.0

    agent.learn_from_game(reward, game)
    print(f"  Result: {outcome}  |  {move_count} moves  |  "
          f"{elapsed:.1f}s  |  patterns known: {len(agent.pattern_weights)}",
          flush=True)

print(f"\nDone. Total games learned: {agent.games_played}  |  "
      f"Patterns: {len(agent.pattern_weights)}", flush=True)