from dataset import ChainsDatasetSV, ChainsDataset
from utils import split_speakers, test_sv
from model import SVModel, AAMSoftmax, SVDummyModel
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
    model = SVModel().to(device)
    model.eval()
    with open("eval.txt", "w") as f: 
        for seed in [43, 33, 30, 20]:
            train_speakers, test_speakers = split_speakers(root_dir = solo_path, train_ratio=train_ratio, seed=seed)
            f.write("test speakers:", len(test_speakers))
            test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            f.write(f"SEED: {seed}, {test_speakers} \n")
            for mode in ["neutral-neutral", "whisper-whisper", "all"]:
                eer = test_sv(test_loader, model, mode, seed, device)
                f.write(f"{mode}: {(eer*100):.2f}%\n")




if __name__ == "__main__":
    main()