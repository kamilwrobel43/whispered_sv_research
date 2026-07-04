from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataConfig:
    solo_path: str = ""
    whsp_path: str = ""
    split_ratio: float = 0.6

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    lr: float = 1e-4

@dataclass
class ModelConfig:
    name: str = "default_model"

@dataclass
class WandbConfig:
    project_name: str = "whispered_sv"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)