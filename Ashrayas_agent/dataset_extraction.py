import numpy as np

# Load the compressed file
loaded_data = np.load("gomoku_dataset_full.npz")

#Player 1 

# Extract X and y using the names we gave them when saving
X = loaded_data["features"]
y = loaded_data["labels"]

print("Dataset loaded!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")



#print(X[58])
#print(y[1])