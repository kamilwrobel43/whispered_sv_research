import torch
import os
from torch.utils.data import Dataset
import torchaudio
from sklearn.utils import shuffle
import soundfile as sf
seed = 43

class ChainsDatasetSV(Dataset):
    def __init__(self, root_solo, root_whsp, speaker_dirs, mode, normalization=True):
        main_dir, second_dir, main_label, second_label = (root_solo, root_whsp, 1, 0) if mode != "whisper" else (root_whsp, root_solo, 0, 1)
        
        self.file_paths = []
        self.speaker_labels = []
        self.labels = []
        self.target_sr = 16000
        self.target_len_s = 4
        self.normalization = normalization

        # solo - 0
        # whsp - 1
        for label, speaker in enumerate(speaker_dirs):
            speaker_dir = os.path.join(main_dir, speaker)
            for file_name in os.listdir(speaker_dir):
                file_path = os.path.join(speaker_dir, file_name)
                self.file_paths.append(file_path)
                self.labels.append(main_label)
                self.speaker_labels.append(label)
                if mode == "both":
                    self.file_paths.append(os.path.join(second_dir, speaker, file_name))
                    self.labels.append(second_label)
                    self.speaker_labels.append(label)

        self.file_paths, self.speaker_labels, self.labels = shuffle(self.file_paths, self.speaker_labels,
                                                                    self.labels, random_state=seed)
    def __len__(self):
        return len(self.file_paths)

    def __load_file__(self, path):
        speech, sr = sf.read(path)
        waveform = torch.from_numpy(speech).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if self.normalization:
            target_rms = 0.1  # arbitrary fixed reference level, just needs to be consistent
            current_rms = waveform.pow(2).mean().sqrt()
            waveform = waveform * (target_rms / (current_rms + 1e-8))
        

        target_len = self.target_len_s * self.target_sr
        if waveform.size(1) < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.size(1)))
        else:
            waveform = waveform[:, :target_len]

        return waveform.squeeze(0)

    def __getitem__(self, idx):

        path = self.file_paths[idx]
        style_label = self.labels[idx]
        speaker_label = self.speaker_labels[idx]

        waveform = self.__load_file__(path)
        return waveform, speaker_label, style_label
    


class ChainsDataset(Dataset):
    def __init__(self, root_solo, root_whsp,  speaker_dirs, normalization=True):
        self.root_dir_solo = root_solo
        self.root_dir_whsp = root_whsp

        self.file_paths_solo = []
        self.file_paths_whsp = []
        self.labels = []
        self.target_sr = 16000
        self.target_len_s = 4
        self.normalization = normalization

        for label, speaker in enumerate(speaker_dirs):
            speaker_dir = os.path.join(self.root_dir_solo, speaker)
            for file_name in os.listdir(speaker_dir):
                file_path = os.path.join(speaker_dir, file_name)
                self.file_paths_solo.append(file_path)
                self.file_paths_whsp.append(os.path.join(self.root_dir_whsp, speaker, file_name))
                self.labels.append(label)

        self.file_paths_solo, self.file_paths_whsp, self.labels = shuffle(self.file_paths_solo,
                                                                          self.file_paths_whsp, self.labels,
                                                                          random_state=seed)

    def __len__(self):
        return len(self.file_paths_solo)

    def __load_file__(self, path):
        speech, sr = sf.read(path)
        waveform = torch.from_numpy(speech).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if self.normalization:
            target_rms = 0.1  # arbitrary fixed reference level, just needs to be consistent
            current_rms = waveform.pow(2).mean().sqrt()
            waveform = waveform * (target_rms / (current_rms + 1e-8))
        # --------------------------------

        target_len = self.target_len_s * self.target_sr
        if waveform.size(1) < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.size(1)))
        else:
            waveform = waveform[:, :target_len]

        return waveform.squeeze(0)

    def __getitem__(self, idx):

        path_solo = self.file_paths_solo[idx]
        path_whsp = self.file_paths_whsp[idx]
        label = self.labels[idx]

        solo = self.__load_file__(path_solo)
        whsp = self.__load_file__(path_whsp)

        return solo, whsp, label
    




