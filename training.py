import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils import sample_triplets, triplet_loss, generate_embeddings, evaluate_modes
from tqdm import tqdm
import wandb






def train_epoch(train_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, optimizer: torch.optim.Optimizer, gamma: float, device = torch.device("cuda" if torch.cuda.is_available else "cpu"), use_amp: bool = False, scaler: GradScaler | None = None):
    
    total_loss, total_loss_trip, total_loss_ce, total_samples = 0.0, 0.0, 0.0, 0
    for solo, whsp, label in tqdm(train_loader, desc="Training"):
        solo = solo.to(device, non_blocking=True)
        whsp = whsp.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            anchors, positives, negatives = sample_triplets(solo, whsp, label, model)
            loss_trip = triplet_loss(anchors, positives, negatives)

            logits = speaker_head(positives, label)
            loss_ce = F.cross_entropy(logits, label)
            loss = loss_trip + gamma * loss_ce

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        curr_batch_size = len(label)
        total_samples += curr_batch_size
        total_loss += loss.item()
        total_loss_trip += loss_trip.item()
        total_loss_ce += loss_ce.item()

    return total_loss / total_samples, total_loss_trip / total_samples, total_loss_ce / total_samples



from omegaconf import ListConfig

def train_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, optimizer: torch.optim.Optimizer, gamma: float, eval_modes: list, n_epochs: int, wandb_project_name: str, wandb_config: dict, unfreezing_schedule: dict[int, list[str]] | None = None, use_amp: bool = False, device = torch.device("cuda" if torch.cuda.is_available else "cpu"), seed: int = 43):
    speaker_head.train()
    if unfreezing_schedule is not None:
        unfreezing_schedule = {
            int(epoch): [str(x) for x in names] if isinstance(names, (list, ListConfig)) else [str(names)]
            for epoch, names in unfreezing_schedule.items()
        }
    else:
        unfreezing_schedule = {}

    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        for epoch in range(1, n_epochs+1):
            if epoch in unfreezing_schedule:
                layer_names = unfreezing_schedule.pop(epoch)
                for name, param in model.named_parameters():
                    if any(target == name or target in name for target in layer_names):
                        if not param.requires_grad:
                            param.requires_grad = True
                            

            scaler = GradScaler(enabled=use_amp)
            model.train()
            train_loss, train_loss_trip, train_loss_ce = train_epoch(train_loader, model, speaker_head, optimizer, gamma, device, use_amp, scaler)
            run.log({"train_loss": train_loss, "train_loss_trip": train_loss_trip, "train_loss_ce": train_loss_ce}, step=epoch)
            model.eval()
            embeddings, speaker_labels, style_labels = generate_embeddings(
                test_loader=test_loader,
                model=model,
                device=device,
                use_amp=use_amp,
            )
            mode_results = evaluate_modes(
                embeddings=embeddings,
                speaker_labels=speaker_labels,
                style_labels=style_labels,
                modes=eval_modes,
                seed=seed,
            )
            for mode, result in mode_results.items():
                run.log({f"{mode}_eval_eer": result}, step=epoch)

            

