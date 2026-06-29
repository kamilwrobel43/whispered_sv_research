from dataset import ChainsDatasetSV, ChainsDataset
from utils import split_speakers
from model import SVModel, AAMSoftmax
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config
from training import train_model

cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    device = torch.device("cpu")
    # root_solo = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/"
    # root_whsp = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/whsp/"

    root_path = cfg.data.root_path
    solo_path = cfg.data.solo_path
    whsp_path = cfg.data.whsp_path
    train_ratio = cfg.data.split_ratio
    batch_size = cfg.training.batch_size



    
    train_speakers, test_speakers = split_speakers(root_dir = solo_path, train_ratio=train_ratio)

    # train_dataset = ChainsDatasetSV(solo_path, whsp_path, train_speakers)
    # test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)

    train_dataset = ChainsDataset(solo_path, whsp_path, train_speakers)
    test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = SVModel().to(device)
    speaker_head = AAMSoftmax(emb_dim=192, n_speakers=len(train_speakers), scale=30.0, margin=0.2).to(device)
    optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "lr": 1e-4},
    {"params": speaker_head.parameters(), "lr": 1e-3} 
])
    
    train_model(train_loader, test_loader, model, speaker_head, optimizer, 0.01, ["all"], 5, device)





if __name__ == "__main__":
    main()