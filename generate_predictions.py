import os
import csv
import random
import torch
import torchaudio
import torchaudio.transforms as T
import kagglehub
from src.model.lcnn import LightCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "saved/lcnn_lfcc_longtrain/model_best.pth"

DATASET_PATH = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
PROTOCOLS_DIR = os.path.join(DATASET_PATH, "LA/LA/ASVspoof2019_LA_cm_protocols")

DEV_PROTOCOL = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.dev.trl.txt")
DEV_AUDIO_DIR = os.path.join(DATASET_PATH, "LA/LA/ASVspoof2019_LA_dev/flac")
EVAL_PROTOCOL = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")
EVAL_AUDIO_DIR = os.path.join(DATASET_PATH, "LA/LA/ASVspoof2019_LA_eval/flac")

SOLUTIONS_DIR = "students_solutions"
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(SOLUTIONS_DIR, "vtdazhy.csv") 

max_len = 64000
N_CALIB = 300


def fit_length(waveform, max_len):
    n = waveform.shape[0]
    if n < max_len:
        return torch.nn.functional.pad(waveform, (0, max_len - n))
    return waveform[:max_len]


def load_model(checkpoint_path):
    model = LightCNN(num_classes=2).to(DEVICE)
    dummy_input = torch.randn(2, 1, 60, 401).to(DEVICE)
    _ = model(dummy_input)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    result = model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    print(checkpoint_path, "->", result)
    return model


lfcc_transform = T.LFCC(
    sample_rate=16000,
    n_lfcc=60,
    speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160}
)


def compute_probs0(model, audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = fit_length(waveform.squeeze(0), max_len)
    lfcc_feat = lfcc_transform(waveform).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(lfcc_feat)["logits"], dim=-1)
    return probs[0, 0].item()


def calibrate_orientation(model):
    bona_paths, spoof_paths = [], []
    with open(DEV_PROTOCOL, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            audio_name, key = parts[1], parts[4]
            path = os.path.join(DEV_AUDIO_DIR, audio_name + ".flac")
            (bona_paths if key == "bonafide" else spoof_paths).append(path)

    random.seed(0)
    bona_sample = random.sample(bona_paths, min(N_CALIB, len(bona_paths)))
    spoof_sample = random.sample(spoof_paths, min(N_CALIB, len(spoof_paths)))

    mean_bona = sum(compute_probs0(model, p) for p in bona_sample) / len(bona_sample)
    mean_spoof = sum(compute_probs0(model, p) for p in spoof_sample) / len(spoof_sample)

    use_index_0 = mean_bona > mean_spoof
    return use_index_0


model = load_model(CHECKPOINT_PATH)
use_index_0 = calibrate_orientation(model)
idx = 0 if use_index_0 else 1

results = []

with open(EVAL_PROTOCOL, "r") as f:
    lines = f.readlines()

with torch.no_grad():
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            audio_name = parts[1]
            waveform, sr = torchaudio.load(os.path.join(EVAL_AUDIO_DIR, audio_name + ".flac"))
            waveform = fit_length(waveform.squeeze(0), max_len)
            lfcc_feat = lfcc_transform(waveform).unsqueeze(0).unsqueeze(0).to(DEVICE)

            probs = torch.softmax(model(lfcc_feat)["logits"], dim=-1)
            score = probs[0, idx].item()
            results.append([audio_name, score])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)
