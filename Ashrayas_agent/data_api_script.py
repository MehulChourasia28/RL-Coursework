from datasets import load_dataset
import numpy as np

print("Loading complete_games.json...")
dataset = load_dataset("json", data_files="https://huggingface.co/datasets/Karesis/Gomoku/resolve/main/gomoku_dataset_split/train/sequence/complete_games.json", split="train")

all_states = []
all_targets = []

print("Processing all turns inside every game...")

for row_data in dataset:
    # 1. Extract the full sequence for this specific game
    move_sequence = []
    for i in range(200):
        move = row_data.get(str(i))
        if move is not None:
            move_sequence.append(move)
            
    if len(move_sequence) < 2:
        continue
        
    # 2. Start with an empty board for this game
    board = np.zeros((15, 15), dtype=np.int8) 
    
    # 3. Play through the game turn-by-turn
    # We stop 1 move before the end, because the last move is a target, not a state
    for step in range(len(move_sequence) - 1):
        
        # Apply the current move to the board
        current_move = move_sequence[step]
        player = 1 if current_move > 0 else 2
        pos = abs(current_move)
        row, col = divmod(pos, 15)
        board[row, col] = player
        
        # The target is whatever move happens NEXT
        target_move = move_sequence[step + 1]
        target_pos = abs(target_move)
        
        # Save a COPY of the board state and the target
        # (Using np.copy is crucial here so we don't overwrite previous states)
        all_states.append(np.copy(board))
        all_targets.append(target_pos)

# 4. Convert to final arrays
X = np.array(all_states)
y = np.array(all_targets)

print("\n--- FULL DATASET READY ---")
print(f"X shape (Features): {X.shape}") 
print(f"y shape (Labels):   {y.shape}")

print("\nSaving dataset locally...")
np.savez_compressed("gomoku_dataset_full.npz", features=X, labels=y)
print("Saved successfully as 'gomoku_dataset_full.npz'!")