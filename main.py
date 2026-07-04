from dataset import ChainsDatasetSV, ChainsDataset
from utils import split_speakers, test_sv
from model import SVModel, AAMSoftmax
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config
from training import train_model
import wandb


cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):


    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # root_solo = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/"
    # root_whsp = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/whsp/"

    solo_path = cfg.data.solo_path
    whsp_path = cfg.data.whsp_path
    train_ratio = cfg.data.split_ratio
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs

    wandb.login()
    wandb_project = cfg.wandb.project_name
    wandb_config = {"epochs": epochs}
    



    
    train_speakers, test_speakers = split_speakers(root_dir = solo_path, train_ratio=train_ratio)

    # train_dataset = ChainsDatasetSV(solo_path, whsp_path, train_speakers)
    # test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)

    train_dataset = ChainsDataset(solo_path, whsp_path, train_speakers)
    test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = SVModel().to(device)
    for param in model.parameters():
        param.requires_grad = False
    speaker_head = AAMSoftmax(emb_dim=192, n_speakers=len(train_speakers), scale=30.0, margin=0.2).to(device)
    optimizer = torch.optim.AdamW([
    {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": 1e-4},
    {"params": speaker_head.parameters(), "lr": 1e-3} 
])
    

    #train_model(train_loader, test_loader, model, speaker_head, optimizer, 0.01, ["all"], epochs, wandb_project, wandb_config, device)
    with open("eval.txt", "w") as f: 
        for seed in [43, 33, 30, 20]:
            train_speakers, test_speakers = split_speakers(root_dir = solo_path, train_ratio=train_ratio, seed=seed)
            test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            f.write(f"SEED: {seed}")
            for mode in ["neutral-neutral", "whisper-whisper", "all"]:
                eer = test_sv(test_loader, model, mode, seed, device)
                f.write(f"{mode}: {(eer*100):.2f}% | ")




if __name__ == "__main__":
    main()