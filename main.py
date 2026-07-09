import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config

import torch
from torch.utils.data import DataLoader

from training import train_model
from dataset import ChainsDataset, ChainsDatasetSV
from utils import split_speakers
from model import SVModel, PostProcessor, CosineSoftmax, AAMSoftmax

cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model.name
    n_epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    lr_ft = cfg.training.lr_ft
    weight_decay = cfg.training.weight_decay
    optimizer_name = cfg.training.optimizer
    seed = cfg.training.seed
    gamma = cfg.training.gamma
    eval_modes = cfg.training.eval_modes
    unfreezing_schedule = cfg.training.unfreezing_schedule

    solo_dir = cfg.data.solo_path
    whsp_dir = cfg.data.whsp_path
    train_ratio = cfg.data.split_ratio

    wandb_project = cfg.wandb.project_name

    wandb_config = {
        "base_model": model_name,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "weight_decay": weight_decay
    }

    train_speakers, test_speakers = split_speakers(root_dir=solo_dir, train_ratio=train_ratio, seed=seed)
    train_dataset = ChainsDataset(solo_dir, whsp_dir, train_speakers)
    test_dataset = ChainsDatasetSV(solo_dir, whsp_dir, test_speakers)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model = PostProcessor(model_name).to(device)
    speaker_head = CosineSoftmax(192, len(train_speakers)).to(device)

    model.base_model.requires_grad_=False
    
    base_params = list(model.base_model.parameters())
    other_params = [p for p in model.parameters() if id(p) not in {id(x) for x in base_params}]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": base_params, "lr": lr_ft},
        {"params": speaker_head.parameters(), "lr": lr},
    ], weight_decay=weight_decay)
    
    train_model(
        train_loader,
        test_loader,
        model,
        speaker_head,
        optimizer,
        gamma,
        eval_modes,
        n_epochs,
        wandb_project,
        wandb_config,
        unfreezing_schedule,
        device,
        seed,
    )

    torch.save(model.state_dict(), "model.pth")







if __name__ == "__main__":
    main()