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
    lr_ft: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 43
    gammas: tuple[float] = (1e-4, 0.5)
    use_amp: bool = True
    eval_modes: list[str] = field(default_factory=lambda: ["neutral-neutral", "neutral-whisper", "whisper-whisper", "all"])
    unfreezing_schedule: dict[int, list[str]] = field(default_factory=lambda: {
        30: ["backbone.stage0"],
        25: ["backbone.stage1"],
        20: ["backbone.stage2"],
        15: ["backbone.stage3"],
        10: ["backbone.stage4"],
        5: ["backbone.stage5"],
        1: [
            "backbone.fin_wght1d.w",
            "pool.linear1.weight",
            "pool.linear1.bias",
            "pool.linear2.weight",
            "pool.linear2.bias",
            "bn.weight",
            "bn.bias",
            "linear.weight",
            "linear.bias",
        ],
    })
    device: str = "cuda"
    speaker_head_name: str = "AAMSofmax"
    speaker_head_scale: float = 30.0
    speaker_head_margin: float = 0.2

@dataclass
class ModelConfig:
    name: str = "default_model"

@dataclass
class WandbConfig:
    project_name: str = "whispered_sv"

@dataclass
class BaselineConfig:
    model_names: list[str] = field(default_factory=lambda: ["redimnet-b6", "redimnet-b2"])
    results_filename: str = "results.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)