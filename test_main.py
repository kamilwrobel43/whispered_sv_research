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
from model import SVModel



batch_size = 128

seeds = [43]


eer_nw = []
eer_nn = []
eer_ww = []
eer_aa = []
f = open("magda.txt", "w")
# Loop through all seeds
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SVModel().to(device)
model.eval()
for seed in seeds:
    f.write(f"seed:  {seed} \n")
    train_speakers, test_speakers = split_speakers(root_dir = "/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/", train_ratio=0.6, seed=seed)
    f.write(f"NUM TEST SPEAKERS: {len(test_speakers)} \n")
    test_dataset = ChainsDatasetSV(root_solo="/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/solo/", root_whsp="/lustre/pd01/hpc-maggol5711-1768234235/datasets/chains/whsp/", speaker_dirs=test_speakers)
    f.write(f"\nTesting dataset: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    f.write(f"TEST SPEAKERS:  {test_speakers} \n")

    
    # Final testing Normal vs. Whispered
    eer = test_sv(test_loader=test_loader, model=model, mode="neutral-whisper", seed=seed)
    eer_nw.append(eer)

    # Final testing Normal vs. Normal
    eer = test_sv(test_loader=test_loader, model=model, mode="neutral-neutral", seed=seed)
    eer_nn.append(eer)

    # Final testing Whispred vs. Whipsered
    eer = test_sv(test_loader=test_loader, model=model, mode="whisper-whisper", seed=seed)
    eer_ww.append(eer)

    # Final testing All vs. All
    eer = test_sv(test_loader=test_loader, model=model, mode="all", seed=seed)
    eer_aa.append(eer)

f.write("#############################  RESULTS  ##############################")
dict = {'NORMAL VS. WHSP': eer_nw,
        'NORMAL VS. NORMAL': eer_nn,
        'WHSP VS. WHSP': eer_ww,
        'ALL VS. ALL': eer_aa}

df = pd.DataFrame(dict)

f.write(str(df))
