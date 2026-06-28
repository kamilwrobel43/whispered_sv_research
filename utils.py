import random
import torch
import os
import numpy as np
from sklearn.metrics import roc_curve
import torch.nn.functional as F


def split_speakers(root_dir='', train_ratio=0.7, seed=43):
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





def generate_balanced_pairs(labels, style_labels, neutral=0, mode="all", seed=42):
    random.seed(seed)

    label_to_indices = {}
    for idx, label_idx in enumerate(labels):
        label_to_indices.setdefault(label_idx, []).append(idx)

    pairs = []
    num_samples = len(labels)

    for i in range(num_samples):
        label_i = labels[i]
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
                return i_style == neutral and j_style == neutral
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

        num_samples = len(embeddings)

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

