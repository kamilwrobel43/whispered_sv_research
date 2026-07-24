from torch.utils.data import DataLoader

from dataset import ChainsDataset, ChainsDatasetSV
from utils import split_speakers, sample_triplets, triplet_loss
from model import SVModel, GRLStyleClassifier, PostProcessor, AAMSoftmax
import torch.nn.functional as F
import torch



if __name__ == "__main__":
    seed = 43
    solo_dir = '/home/kamil/Datasets/chains/solo'
    whsp_dir = '/home/kamil/Datasets/chains/whsp'


    train_speakers, test_speakers = split_speakers(root_dir=solo_dir, train_ratio=0.7, seed=seed)
    
    train_dataset = ChainsDataset(solo_dir, whsp_dir, train_speakers)
    test_dataset = ChainsDatasetSV(solo_dir, whsp_dir, test_speakers)
    train_loader = DataLoader(train_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    postprocessor = PostProcessor()
    grl_head = GRLStyleClassifier(64)
    speaker_head = AAMSoftmax(192, n_speakers=len(train_speakers), scale=30, margin=0.3)

    
    for solo, whsp, label in train_loader:

        solo_enc, solo = postprocessor.encode(solo)
        whsp_enc, whsp = postprocessor.encode(whsp)
        anch, pos, neg = sample_triplets(solo_enc, solo, whsp_enc, whsp, label, postprocessor)
        loss_trip = triplet_loss(anch, pos, neg)

        solo_logits = grl_head(solo_enc)
        whsp_logits = grl_head(whsp_enc)

        loss_adv = (F.cross_entropy(solo_logits, torch.ones_like(label)) + F.cross_entropy(whsp_logits, torch.zeros_like(label))) // 2

        logits = speaker_head(pos, label)

        loss_ce = F.cross_entropy(logits, label)

        print("DONE")
        break


