from dataclasses import asdict, dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "gomoku_policy_value.pt"


@dataclass(slots=True)
class NetworkConfig:
    board_size: int = 15
    input_planes: int = 4
    channels: int = 96
    residual_blocks: int = 6
    value_hidden_dim: int = 128


@dataclass(slots=True)
class SearchConfig:
    simulations: int = 48
    c_puct: float = 1.6
    dirichlet_alpha: float = 0.08
    dirichlet_fraction: float = 0.20
    temperature_moves: int = 10
    exploration_temperature: float = 1.10
    deterministic_temperature: float = 1e-3
    heuristic_prior_blend: float = 0.18


@dataclass(slots=True)
class TrainingConfig:
    iterations: int = 60
    bootstrap_games: int = 120
    self_play_games_per_iteration: int = 24
    evaluation_games: int = 16
    max_moves_per_game: int = 225
    batch_size: int = 128
    batches_per_iteration: int = 180
    replay_buffer_size: int = 12000
    learning_rate: float = 2.5e-4
    weight_decay: float = 1.0e-4
    value_loss_weight: float = 1.0
    grad_clip_norm: float = 1.0
    checkpoint_path: str = str(DEFAULT_CHECKPOINT)
    seed: int = 7


@dataclass(slots=True)
class RuntimeConfig:
    checkpoint_path: str = str(DEFAULT_CHECKPOINT)
    board_size: int = 15
    device: str = "auto"
    use_search: bool = True
    simulations: int = 56
    heuristic_fallback: bool = True


def default_training_config() -> TrainingConfig:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return TrainingConfig()


def default_runtime_config() -> RuntimeConfig:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return RuntimeConfig()


def dataclass_to_dict(config):
    return asdict(config)