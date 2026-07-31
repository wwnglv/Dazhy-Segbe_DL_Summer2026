import os
import torch
import torchaudio
import torchaudio.transforms as T
from src.datasets.base_dataset import BaseDataset

class VoiceDataset(BaseDataset):
    def __init__(self, protocol_path, audio_dir, max_len=64000, sample_rate=16000, n_lfcc=60, **kwargs):
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.sample_rate = sample_rate

        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": 512,
                "win_length": 400,
                "hop_length": 160,
            }
        )

        index = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    audio_name = parts[1]
                    key = parts[4]
                    label = 1 if key == "bonafide" else 0
                    audio_path = os.path.join(self.audio_dir, audio_name + ".flac")

                    index.append({
                        "path": audio_path,
                        "label": label
                    })

        super().__init__(index, **kwargs)

    def load_object(self, path):
        waveform, sr = torchaudio.load(path)
        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.max_len:
            pad_len = self.max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:self.max_len]

        lfcc_features = self.lfcc_transform(waveform).unsqueeze(0)
        return lfcc_features
