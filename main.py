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
import pandas as pd

cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):

    solo_path = cfg.data.solo_path
    whsp_path = cfg.data.whsp_path
    split_ratio = cfg.data.split_ratio
    batch_size = cfg.training.batch_size

    eer_nw = []
    eer_nn = []
    eer_ww = []
    eer_aa = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVModel().to(device)
    model.eval()
    with open("kamil.txt", "w") as f: 
        for seed in [43, 33, 30, 20]:
            _, test_speakers = split_speakers(root_dir = solo_path, train_ratio=split_ratio, seed=seed)
            f.write(f"NUM TEST SPEAKERS: {len(test_speakers)} \n")
            test_dataset = ChainsDatasetSV(solo_path, whsp_path, test_speakers)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            f.write(f"SEED: {seed}, {test_speakers} \n")
            
            # Final testing Normal vs. Whispered
            eer = test_sv(test_loader=test_loader, model=model, mode="neutral-whisper", seed=seed)
            eer_nw.append(eer)

            # Final testing Normal vs. Normal
            eer = test_sv(test_loader=test_loader, model=model, mode="neutral-neutral", seed=seed)
            eer_nn.append(eer)

            # Final testing Whispred vs. Whipsered
            eer = test_sv(test_loader=test_loader, model=model, mode="whisper-whisper", seed=seed)
            eer_ww.append(eer)

            # Final testing All vs. All
            eer = test_sv(test_loader=test_loader, model=model, mode="all", seed=seed)
            eer_aa.append(eer)

        f.write("#############################  RESULTS  ##############################")
        dict = {'NORMAL VS. WHSP': eer_nw,
                'NORMAL VS. NORMAL': eer_nn,
                'WHSP VS. WHSP': eer_ww,
                'ALL VS. ALL': eer_aa}

        df = pd.DataFrame(dict)
        f.write(str(df))


if __name__ == "__main__":
    main()