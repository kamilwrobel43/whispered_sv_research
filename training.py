import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omegaconf import ListConfig
from utils import sample_triplets, triplet_loss, generate_embeddings, evaluate_modes
from tqdm import tqdm
import wandb



def train_epoch_no_triplet(train_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, style_head: nn.Module, optimizer: torch.optim.Optimizer, gammas: tuple[float], device: torch.device, use_amp: bool, scaler: GradScaler | None, grl_scheduler):
    total_loss, total_loss_ce, total_loss_adv, total_samples = 0.0, 0.0, 0.0, 0

    for waveform, speaker_label, style_label in tqdm(train_loader, desc="Training"):
        if grl_scheduler:
            curr_alpha = grl_scheduler.step()
            style_head.grl.alpha = torch.tensor(
                [curr_alpha],
                dtype=torch.float32,
                device=device,
                requires_grad=False
            )

        waveform = waveform.to(device)
        speaker_label = speaker_label.to(device)
        style_label = style_label.to(device)

        with autocast(device.type, enabled=use_amp):

            emb_enc, emb = model.encode(waveform)
            style_logits = style_head(emb_enc)

            loss_adv = F.cross_entropy(style_logits, style_label)

            emb_dec = model.decode(emb_enc, emb)
            speaker_logits = speaker_head(emb_dec)
            loss_ce = F.cross_entropy(speaker_logits, speaker_label)

            loss = gammas[0]*loss_ce + gammas[1]*loss_adv

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        curr_batch_size = len(speaker_label)
        total_samples += curr_batch_size
        total_loss += loss.item()
        total_loss_ce += loss_ce.item()
        total_loss_adv += loss_adv.item()

    return total_loss / total_samples, total_loss_ce / total_samples, total_loss_adv / total_samples


def train_model_no_triplet(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, style_head: nn.Module, optimizer: torch.optim.Optimizer, gammas: tuple[float], eval_modes: list, n_epochs: int, wandb_project_name: str, wandb_config: dict, unfreezing_schedule: dict[int, list[str]] | None = None, use_amp: bool = False, device = torch.device("cuda" if torch.cuda.is_available else "cpu"), seed: int = 43, scheduler: torch.optim.lr_scheduler.LRScheduler | None = None, grl_scheduler = None):
    speaker_head.train()
    style_head.train()

    if unfreezing_schedule is not None:
        unfreezing_schedule = {
            int(epoch): [str(x) for x in names] if isinstance(names, (list, ListConfig)) else [str(names)]
            for epoch, names in unfreezing_schedule.items()
        }
    else:
        unfreezing_schedule = {}
    
    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        scaler = GradScaler(enabled=use_amp)
        for epoch in range(1, n_epochs+1):
            if epoch in unfreezing_schedule:
                layer_names = unfreezing_schedule.pop(epoch)
                for name, param in model.named_parameters():
                    if any(target == name or target in name for target in layer_names):
                        if not param.requires_grad:
                            param.requires_grad = True
    
    
            model.train()
            train_loss, train_loss_ce, train_loss_adv = train_epoch_no_triplet(
                train_loader,
                model,
                speaker_head,
                style_head,
                optimizer,
                gammas,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                grl_scheduler=grl_scheduler,
            )
            run.log({"train_loss": train_loss,"train_loss_ce": train_loss_ce, "train_loss_adv": train_loss_adv}, step=epoch)
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
    
            if scheduler is not None:
                scheduler.step()



def train_epoch(train_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, style_head: nn.Module, optimizer: torch.optim.Optimizer, gammas: tuple[float], device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), use_amp: bool = False, scaler: GradScaler | None = None, grl_scheduler = None):
    
    total_loss, total_loss_trip, total_loss_ce, total_loss_adv, total_samples = 0.0, 0.0, 0.0, 0.0, 0
    for solo, whsp, label in tqdm(train_loader, desc="Training"):

        if grl_scheduler:
            curr_alpha = grl_scheduler.step()
            style_head.grl.alpha = torch.tensor(
                [curr_alpha],
                dtype=torch.float32,
                device=device,
                requires_grad=False
            )
        solo = solo.to(device, non_blocking=True)
        whsp = whsp.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        device_type = device.type if isinstance(device, torch.device) else str(device)
        with autocast(device_type, enabled=use_amp):

            solo_enc, solo = model.encode(solo)
            whsp_enc, whsp = model.encode(whsp)
            anch, pos, neg = sample_triplets(solo_enc, solo, whsp_enc, whsp, label, model)
            loss_trip = triplet_loss(anch, pos, neg)

            solo_logits = style_head(solo_enc)
            whsp_logits = style_head(whsp_enc)

            loss_adv = (F.cross_entropy(solo_logits, torch.ones_like(label)) + F.cross_entropy(whsp_logits, torch.zeros_like(label))) / 2

            logits = speaker_head(pos, label)

            loss_ce = F.cross_entropy(logits, label)

            loss = loss_trip + gammas[0]*loss_ce + gammas[1]*loss_adv

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
        total_loss_adv += loss_adv.item()

    return total_loss / total_samples, total_loss_trip / total_samples, total_loss_ce / total_samples, total_loss_adv / total_samples





def train_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, style_head: nn.Module, optimizer: torch.optim.Optimizer, gammas: tuple[float], eval_modes: list, n_epochs: int, wandb_project_name: str, wandb_config: dict, unfreezing_schedule: dict[int, list[str]] | None = None, use_amp: bool = False, device = torch.device("cuda" if torch.cuda.is_available else "cpu"), seed: int = 43, scheduler: torch.optim.lr_scheduler.LRScheduler | None = None, grl_scheduler = None):
    speaker_head.train()
    style_head.train()

    if unfreezing_schedule is not None:
        unfreezing_schedule = {
            int(epoch): [str(x) for x in names] if isinstance(names, (list, ListConfig)) else [str(names)]
            for epoch, names in unfreezing_schedule.items()
        }
    else:
        unfreezing_schedule = {}

    with wandb.init(project=wandb_project_name, config=wandb_config) as run:
        scaler = GradScaler(enabled=use_amp)
        for epoch in range(1, n_epochs+1):
            if epoch in unfreezing_schedule:
                layer_names = unfreezing_schedule.pop(epoch)
                for name, param in model.named_parameters():
                    if any(target == name or target in name for target in layer_names):
                        if not param.requires_grad:
                            param.requires_grad = True
                            

            model.train()
            train_loss, train_loss_trip, train_loss_ce, train_loss_adv = train_epoch(
                train_loader,
                model,
                speaker_head,
                style_head,
                optimizer,
                gammas,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                grl_scheduler=grl_scheduler,
            )
            run.log({"train_loss": train_loss, "train_loss_trip": train_loss_trip, "train_loss_ce": train_loss_ce, "train_loss_adv": train_loss_adv}, step=epoch)
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

            if scheduler is not None:
                scheduler.step()

            

