"""Central configuration for Gomoku settings."""

# Change this single value to resize the board across Gomoku modules.
BOARD_SIZE = 15

# Visual scaling for PyGame board rendering.
# The board window will target at most this many pixels in width/height.
MAX_BOARD_PIXELS = 650

# Cell size constraints used by automatic scaling.
DEFAULT_CELL_SIZE = 60
MIN_CELL_SIZE = 28
