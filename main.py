import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config

import torch
from torch.utils.data import DataLoader

from training import train_model
from dataset import ChainsDataset, ChainsDatasetSV
from utils import split_speakers
from model import SVModel, CosineSoftmax, AAMSoftmax

cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model.name
    n_epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    optimizer_name = cfg.training.optimizer
    seed = cfg.training.seed
    gamma = cfg.training.gamma
    eval_modes = cfg.training.eval_modes

    solo_dir = cfg.data.solo_path
    whsp_dir = cfg.data.whsp_path
    train_ratio = cfg.data.split_ratio

    wandb_project = cfg.wandb.project_name

    wandb_config = {
        "base_model": model_name,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr
    }


    train_speakers, test_speakers = split_speakers(root_dir=solo_dir, train_ratio=train_ratio, seed=seed)
    train_dataset = ChainsDataset(solo_dir, whsp_dir, train_speakers)
    test_dataset = ChainsDatasetSV(solo_dir, whsp_dir, test_speakers)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = SVModel(model_name).to(device)
    for name, param in model.named_parameters():
        if "stage4" in name or "stage5" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    unfrozen_params = [p for p in model.parameters() if p.requires_grad]

    speaker_head = CosineSoftmax(emb_dim=192, n_speakers=len(train_speakers)).to(device)
    optimizer = torch.optim.Adam([{'params': unfrozen_params, 'lr': lr},
                                   {'params': speaker_head.parameters(), 'lr': lr}])

    
    
    train_model(train_loader, test_loader, model, speaker_head, optimizer, gamma, eval_modes, n_epochs, wandb_project, wandb_config, device, seed)

    torch.save(model.state_dict(), "model.pth")







if __name__ == "__main__":
    main()