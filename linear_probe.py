import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import WavLMModel

from dataset import ChainsDatasetSV
from utils import split_speakers

import hydra
from hydra.core.config_store import ConfigStore
from configs.config_classes import Config



def linear_probe(model, train_loader, test_loader, wandb_config, n_layers, n_epochs, device, speaker_mode, n_classes, emb_dim, probe_mode, model_name):
    for layer in range(n_layers):
        classifier = nn.Linear(emb_dim, n_classes).to(device)
        optimizer = torch.optim.Adam(lr=1e-3, params=classifier.parameters())
        with wandb.init(project="whispered_sv", config=wandb_config, name=f"{probe_mode}_probe_{model_name}_{layer}", reinit=True) as run:
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

                    if model_name == "wavlm-base":
                        outputs = model(input_values = waveform, output_hidden_states=True)
                        features = outputs.hidden_states[layer].mean(dim=1)
                    else:
                        features = model(waveform)

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

                        if model_name == "wavlm-base":
                            outputs = model(input_values = waveform, output_hidden_states=True)
                            features = outputs.hidden_states[layer].mean(dim=1)
                        else:
                            features = model(waveform)

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
    root_solo = cfg.data.solo_path
    root_whsp = cfg.data.whsp_path
    seed = cfg.training.seed
    split_ratio = cfg.data.split_ratio
    probe_mode = cfg.linear_probe.mode
    model_name = cfg.linear_probe.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    n_epochs = cfg.linear_probe.n_epochs

    

    if probe_mode == "speaker-whisper-whisper":
        train_speakers, _ = split_speakers(root_solo, train_ratio=1.0, seed=seed)
        dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "whisper")
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, val_size])
        n_classes = len(train_speakers)
        speaker_mode = True

    elif probe_mode == "speaker-normal-whisper":
        train_speakers, _ = split_speakers(root_solo, train_ratio=1.0, seed=seed)
        train_dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "whisper")
        test_dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "normal")
        n_classes = len(train_speakers)
        speaker_mode = True

    elif probe_mode == "speaker-normal-normal":
        train_speakers, _ = split_speakers(root_solo, train_ratio=1.0, seed=seed)
        dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "normal")
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, val_size])
        n_classes = len(train_speakers)
        n_classes = len(train_speakers)
        speaker_mode = True

    elif probe_mode == "style":
        train_speakers, test_speakers = split_speakers(root_solo, train_ratio=split_ratio, seed=seed)
        train_dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "both")
        test_dataset = ChainsDatasetSV(root_solo, root_whsp, test_speakers, mode = "both")
        n_classes = 2
        speaker_mode = False


    
    if model_name == "wavlm-base":
        model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
        emb_dim = 768
        n_layers = 13
    elif model_name == "redimnet-b6":
        model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                               model_name="b6", 
                               train_type="ft_lm", 
                               dataset="vox2").to(device)
        emb_dim = 192
        n_layers = 1

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    train_loader= DataLoader(train_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



    wandb_config = {
            "base_model": model_name,
            "batch_size": 16,
            "lr": 1e-3,
        }


    linear_probe(model, train_loader, test_loader, wandb_config, n_layers, n_epochs, device, speaker_mode, n_classes, emb_dim, probe_mode, model_name)


if __name__ == "__main__":
    main()