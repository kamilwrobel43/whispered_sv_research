import random
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from tqdm import tqdm




def split_speakers(root_dir='', train_ratio=0.6, seed=43):
    random.seed(seed)
    speakers = sorted(os.listdir(root_dir))
    random.shuffle(speakers)
    total_speakers = len(speakers)
    train_end_idx = int(total_speakers * train_ratio)

    train_speakers = speakers[:train_end_idx]
    test_speakers = speakers[train_end_idx:]

    return train_speakers, test_speakers




def sample_triplets(solo_batch, whsp_batch, labels, model):
    anchors = model(solo_batch)    # shape: [batch_size, embedding_dim]
    positives = model(whsp_batch)  # shape: [batch_size, embedding_dim]

    batch_size = labels.size(0)

    mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    random_noise = torch.rand(batch_size, batch_size, device=labels.device)
    
    sampling_scores = torch.where(mask, random_noise, torch.tensor(-float('inf'), device=labels.device))
    neg_indices = torch.argmax(sampling_scores, dim=1)


    negatives = anchors[neg_indices]

    return anchors, positives, negatives


def triplet_loss(anchors, positives, negatives, margin=0.2):
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)
    return F.triplet_margin_loss(anchors, positives, negatives, margin=margin, p=2)





def generate_balanced_pairs(speaker_labels, style_labels, neutral=0, mode="all", seed=42):
    random.seed(seed)

    label_to_indices = {}
    for idx, speaker_label_idx in enumerate(speaker_labels):
        label_to_indices.setdefault(speaker_label_idx, []).append(idx)

    pairs = []
    num_samples = len(speaker_labels)

    for i in range(num_samples):
        label_i = speaker_labels[i]
        style_i = style_labels[i]

        # Positive pairs (same speaker)
        pos_candidates = [
            j for j in label_to_indices[label_i]
            if j != i
        ]

        # Negative pairs (different speaker)
        neg_candidates = [
            j for lbl in label_to_indices if lbl != label_i
            for j in label_to_indices[lbl]
        ]

        def style_match(i_style, j_style):
            if mode == "neutral-neutral":
                return i_style == j_style == neutral
            elif mode == "whisper-whisper":
                return i_style != neutral and j_style != neutral
            elif mode == "neutral-whisper":
                return (i_style == neutral and j_style != neutral) or (i_style != neutral and j_style == neutral)
            elif mode == "all":
                return True

        # Filter by style rules
        pos_candidates = [j for j in pos_candidates if style_match(style_i, style_labels[j])]
        neg_candidates = [j for j in neg_candidates if style_match(style_i, style_labels[j])]

        if pos_candidates:
            pairs.append((i, random.choice(pos_candidates)))
        if neg_candidates:
            pairs.append((i, random.choice(neg_candidates)))

    return pairs


def compute_eer(embeddings, labels, pairs):
        scores = []
        targets = []

        embeddings = F.normalize(embeddings, p=2, dim=1)
        labels = np.array(labels)


        for i, j in pairs:
            dis = 1 - F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            same = int(labels[i] == labels[j])
            scores.append(dis)
            targets.append(same)

        scores = np.array(scores)
        targets = np.array(targets)

        pos = 0
        neg = 0
        c_pos = 0
        c_neg = 0

        for i in range(len(scores)):
            if targets[i] == 1:
                pos += scores[i]
                c_pos += 1
            else:
                neg += scores[i]
                c_neg += 1

        print('pos', c_pos, pos / c_pos, 'neg', c_neg, neg / c_neg)

        fpr, tpr, thresholds = roc_curve(targets, -scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

        return eer, eer_threshold


def generate_embeddings(test_loader, model, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_embeddings = []
    all_speaker_labels = []
    all_style_labels = []

    with torch.no_grad():
        for feats, label, style_label in tqdm(test_loader, desc="Generating embeddings"):
            feats = feats.to(device)
            label = label.to(device)
            style_label = style_label.to(device)

            emb = model(feats)
            all_embeddings.append(emb.cpu())
            all_speaker_labels.extend(label.cpu().numpy())
            all_style_labels.extend(style_label.cpu().numpy())

    embeddings = torch.cat(all_embeddings)
    return embeddings, np.array(all_speaker_labels), np.array(all_style_labels)


def evaluate_mode(embeddings, speaker_labels, style_labels, mode, seed):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    if isinstance(speaker_labels, list):
        speaker_labels = np.array(speaker_labels)
    if isinstance(style_labels, list):
        style_labels = np.array(style_labels)

    pairs = generate_balanced_pairs(speaker_labels, style_labels, neutral=0, mode=mode, seed=seed)
    eer, _ = compute_eer(embeddings, speaker_labels, pairs)
    return eer


def evaluate_modes(embeddings, speaker_labels, style_labels, modes, seed):
    results = {}
    for mode in modes:
        result = evaluate_mode(embeddings, speaker_labels, style_labels, mode, seed)
        results[mode] = result
    return results


def test_sv(test_loader, model, mode, seed, device=None):
    embeddings, speaker_labels, style_labels = generate_embeddings(test_loader, model, device=device)
    result = evaluate_mode(embeddings, speaker_labels, style_labels, mode, seed)
    return result

