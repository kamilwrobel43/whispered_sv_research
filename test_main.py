import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.optim as optim
import sys
import random
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import sklearn.metrics
from sklearn import preprocessing
from torch.autograd import Function
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import cosine
from torch.nn.utils.rnn import pad_sequence
import math
from utils import *
from dataset import ChainsDatasetSV
import pandas as pd



batch_size = 128
num_epochs = 100
save_interval = 50
dropout_v = 0.3
lr_main = 1e-4
lr_head = 1e-4
lr_ecapa = 1e-5
gamma = 1e-4
aug = 0.5

seeds = [93829758]


eer_nw = []
eer_nn = []
eer_ww = []
eer_aa = []

# Loop through all seeds
for q in range(len(seeds)):
    seed = seeds[q]
    print("################################################################################################################")
    print("seed: ", seed)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    train_speakers, test_speakers = split_speakers(root_dir = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/", train_ratio=0.6)
    print("Train_speakers: ", train_speakers, "\tTest_speakers: ", test_speakers)

    n_speakers = len(train_speakers)
    
    model = torch.hub.load(
        "IDRnD/ReDimNet",
        "ReDimNet",
        model_name="b6",
        train_type="ft_lm",
        dataset="vox2"
    ).to(device)

    # -----------------------------------------------------------------------------------------------
    # Speaker verification testing

    test_dataset = ChainsDatasetSV(root_solo="/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/", root_whsp="/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/whsp/", speaker_dirs=test_speakers)
    print("\nTesting dataset: ", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    # Final testing Normal vs. Whispered
    eer = test_sv(model=model, mode="neutral-whisper", seed=seed)
    eer_nw.append(eer)

    # Final testing Normal vs. Normal
    eer = test_sv(model=model, mode="neutral-neutral", seed=seed)
    eer_nn.append(eer)

    # Final testing Whispred vs. Whipsered
    eer = test_sv(model=model, mode="whisper-whisper", seed=seed)
    eer_ww.append(eer)

    # Final testing All vs. All
    eer = test_sv(model=model, mode="all", seed=seed)
    eer_aa.append(eer)

print("#############################  RESULTS  ##############################")
dict = {'NORMAL VS. WHSP': eer_nw,
        'NORMAL VS. NORMAL': eer_nn,
        'WHSP VS. WHSP': eer_ww,
        'ALL VS. ALL': eer_aa}

df = pd.DataFrame(dict)

print(df)