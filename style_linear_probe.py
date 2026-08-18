from dataset import ChainsDatasetSV
from utils import split_speakers
from transformers import WavLMModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm


def main():
    root_solo = '/home/kamil/Datasets/chains/solo'
    root_whsp = '/home/kamil/Datasets/chains/whsp'
    seed = 43

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)


    classifier = nn.Linear(768, 1).to(device)
    optimizer = torch.optim.Adam(lr=1e-3, params=classifier.parameters())

    train_speakers, test_speakers = split_speakers(root_solo, train_ratio=0.7, seed = seed)
    train_dataset = ChainsDatasetSV(root_solo, root_whsp, train_speakers, mode = "both")
    test_dataset = ChainsDatasetSV(root_solo, root_whsp, test_speakers, mode = "both")

    train_loader= DataLoader(train_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



    n_epochs = 200
    n_layers = 13

    wandb_config = {
            "base_model": "wavlm-base",
            "batch_size": 16,
            "lr": 1e-3,
        }


    for layer in range(n_layers):
        with wandb.init(project="whispered_sv", config=wandb_config, name=f"style_probe_{layer}", reinit=True) as run:
            best_loss = torch.inf
            patience = 0
            for epoch in range(n_epochs):
                total_train_loss = 0
                total_correct = 0
                total_samples = 0
                model.train()
                for waveform, _, label in tqdm(train_loader, desc="Training"):
                    batch_size = waveform.size(0)
                    waveform = waveform.to(device)
                    label = label.unsqueeze(1).to(device).float()
                    outputs = model(input_values = waveform, output_hidden_states=True)

                    features = outputs.hidden_states[layer].mean(dim=1)

                    logits = classifier(features)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)

                    probs = torch.sigmoid(logits)                        
                    preds = (probs > 0.5).long().squeeze(1)
                    labels = label.squeeze(1).long()                     
                    total_correct += (preds == labels).sum().item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss+=loss.item()*batch_size
                    total_samples+=batch_size

                total_train_loss /= total_samples
                train_accuracy = total_correct / total_samples

                model.eval()
                with torch.inference_mode():

                    total_test_loss = 0
                    total_correct = 0
                    total_samples = 0
                    
                    for waveform, _, label in tqdm(test_loader, desc="Evaluating"):
                        waveform = waveform.to(device)
                        batch_size = waveform.size(0)
                        label = label.unsqueeze(1).to(device).float()
                        outputs = model(input_values = waveform, output_hidden_states=True)

                        features = outputs.hidden_states[layer].squeeze(1).mean(dim=1)

                        logits = classifier(features)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)

                        probs = torch.sigmoid(logits)                        
                        preds = (probs > 0.5).long().squeeze(1)
                        labels = label.squeeze(1).long()                     
                        total_correct += (preds == labels).sum().item()
                        
                        total_test_loss+=loss.item()*batch_size
                        total_samples += batch_size
                    total_test_loss /= total_samples
                    test_accuracy = total_correct / total_samples

                    
                run.log({"train_accuracy": train_accuracy, "train_loss": total_train_loss, "test_loss": total_test_loss, "test_accuracy": test_accuracy}, step=epoch)
                if total_test_loss < best_loss:
                    patience = 0
                    best_loss = total_test_loss
                elif patience==15:
                    break




            


    

    

if __name__ == "__main__":
    main()