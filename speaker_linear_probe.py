import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WavLMModel

from dataset import ChainsDatasetSV
from utils import split_speakers

import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config





def linear_probe(model, classifier, train_loader, test_loader, wandb_config, n_layers, n_epochs, device, optimizer, speaker_mode):
    for layer in range(n_layers):
        torch.nn.init.xavier_uniform_(classifier.weight)
        with wandb.init(project="whispered_sv", config=wandb_config, name=f"speaker_probe_{layer}", reinit=True) as run:
            best_loss = torch.inf
            patience = 0
            for epoch in range(n_epochs):
                total_train_loss = 0
                total_correct = 0
                total_samples = 0
                classifier.train()
                for waveform, speaker_label, style_label in tqdm(train_loader, desc="Training"):
                    batch_size = waveform.size(0)
                    waveform = waveform.to(device)
                    label = speaker_label.to(device).long() if speaker_mode else style_label.to(device).long()
                    outputs = model(input_values = waveform, output_hidden_states=True)

                    features = outputs.hidden_states[layer].mean(dim=1)

                    logits = classifier(features)
                    loss = torch.nn.functional.cross_entropy(logits, label)

                    preds = logits.argmax(dim=1)
                    total_correct += (preds == label).sum().item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss+=loss.item()*batch_size
                    total_samples+=batch_size

                total_train_loss /= total_samples
                train_accuracy = total_correct / total_samples

                classifier.eval()
                with torch.inference_mode():

                    total_test_loss = 0
                    total_correct = 0
                    total_samples = 0
                    
                    for waveform, speaker_label, style_label in tqdm(test_loader, desc="Evaluating"):
                        waveform = waveform.to(device)
                        batch_size = waveform.size(0)
                        label = speaker_label.to(device).long() if speaker_mode else style_label.to(device).long()
                        outputs = model(input_values = waveform, output_hidden_states=True)

                        features = outputs.hidden_states[layer].mean(dim=1)

                        logits = classifier(features)
                        loss = torch.nn.functional.cross_entropy(logits, label)

                        preds = logits.argmax(dim=1)
                        total_correct += (preds == label).sum().item()
                        
                        total_test_loss+=loss.item()*batch_size
                        total_samples += batch_size
                    total_test_loss /= total_samples
                    test_accuracy = total_correct / total_samples

                    
                run.log({"train_accuracy": train_accuracy, "train_loss": total_train_loss, "test_loss": total_test_loss, "test_accuracy": test_accuracy}, step=epoch)
                if total_test_loss < best_loss:
                    patience = 0
                    best_loss = total_test_loss
                else:
                    patience+=1

                if patience==15:
                    break

cs = ConfigStore.instance()
cs.store(name="base_cfg", node=Config)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    root_solo = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/"
    root_whsp = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/whsp/"
    seed = 43

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    train_speakers, test_speakers = split_speakers(root_solo, train_ratio=1.0, seed = seed)
    train_dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "normal")
    test_dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "whisper")

    model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    classifier = nn.Linear(768, len(train_speakers)).to(device)
    optimizer = torch.optim.Adam(lr=1e-3, params=classifier.parameters())

    
    train_loader= DataLoader(train_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    n_epochs = 200
    n_layers = 13

    wandb_config = {
            "base_model": "wavlm-base",
            "batch_size": 16,
            "lr": 1e-3,
        }


    linear_probe(model, classifier, train_loader, test_loader, wandb_config, n_layers, n_epochs, device, optimizer, speaker_mode = True)


if __name__ == "__main__":
    main()