import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import sample_triplets, triplet_loss, generate_embeddings, evaluate_modes
from tqdm import tqdm
import wandb






def train_epoch(train_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, optimizer: torch.optim.Optimizer, gamma: float, device = torch.device("cuda" if torch.cuda.is_available else "cpu")):
    
    total_loss, total_loss_trip, total_loss_ce, total_samples = 0.0, 0.0, 0.0, 0
    for solo, whsp, label in tqdm(train_loader, desc="Training"):
        solo, whsp, label = solo.to(device), whsp.to(device), label.to(device)

        anchors, positives, negatives = sample_triplets(solo, whsp, label, model)
        loss_trip = triplet_loss(anchors, positives, negatives)

        logits = speaker_head(positives, label)

        loss_ce = F.cross_entropy(logits, label)
        loss = loss_trip + gamma * loss_ce
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_batch_size = len(label)
        total_samples += curr_batch_size
        total_loss += loss.item()
        total_loss_trip += loss_trip.item()
        total_loss_ce += loss_ce.item()

    return total_loss / total_samples, total_loss_trip / total_samples, total_loss_ce / total_samples



def train_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, optimizer: torch.optim.Optimizer, gamma: float, eval_modes: list, n_epochs: int, wandb_project_name: str, wandb_config: dict, device = torch.device("cuda" if torch.cuda.is_available else "cpu"), seed: int = 43):
    speaker_head.train()
    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        for epoch in range(1, n_epochs+1):

            model.train()
            train_loss, train_loss_trip, train_loss_ce = train_epoch(train_loader, model, speaker_head, optimizer, gamma, device)
            run.log({"train_loss": train_loss, "train_loss_trip": train_loss_trip, "train_loss_ce": train_loss_ce})
            model.eval()
            embeddings, speaker_labels, style_labels = generate_embeddings(
                test_loader=test_loader,
                model=model,
                device=device,
            )
            mode_results = evaluate_modes(
            embeddings=embeddings,
            speaker_labels=speaker_labels,
            style_labels=style_labels,
            modes=eval_modes,
            seed=seed,
        )
        for mode, result in mode_results.items():
            run.log({f"{mode}_eval_eer": result})

            

