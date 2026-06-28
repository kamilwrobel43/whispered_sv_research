import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import sample_triplets, triplet_loss



def train_epoch(train_loader: DataLoader, model: nn.Module, speaker_head: nn.Module, optimizer: torch.optim.Optimizer, gamma: float, device = torch.device("cuda" if torch.cuda.is_available else "cpu")):
    
    total_loss, total_loss_trip, total_loss_ce, total_samples = 0.0, 0.0, 0.0, 0
    for solo, whsp, label in train_loader:
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
        total_loss += loss.item() * curr_batch_size
        total_loss_trip += loss_trip.item() * curr_batch_size
        total_loss_ce += loss_ce.item() * curr_batch_size

    return total_loss / total_samples, total_loss_trip / total_samples, total_loss_ce / total_samples


        


    

