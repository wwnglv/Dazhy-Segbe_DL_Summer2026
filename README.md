
# ASVspoof 2019

A solution for the synthetic speech detection task (Logical Access) on the [ASVspoof 2019](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) dataset, developed as part of the Deep Learning mini-course, Summer 2026. The repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## Results

| Metric | Value |
|---|---|
| EER on eval set | **7.91%** |
| EER on dev set (Epoch 60) | 0.64% |

Full training logs (60 epochs, all metrics): [WandB Report](https://api.wandb.ai/links/wwnglv-hse/5ppquf69)

## Architecture

LCNN (Light CNN) with Max-Feature-Map (MFM) activation:
- Front-end: LFCC, 60 coefficients (`n_fft=512`, `win_length=400`, `hop_length=160`)
- 4 convolutional blocks (1→32→64→128→256 channels) with MFM and MaxPool
- Dropout(p=0.7) → BatchNorm1d(128) → Linear(128, 2)
- Loss: CrossEntropyLoss, Optimizer: Adam (lr=3e-4), CosineAnnealingLR, 60 epochs

Architecture details and a description of the experiments conducted can be found in the report file (`otchet_praktika.docx`) in this repository.

## Installation

```bash
git clone [https://github.com/wwnglv/Dazhy-Segbe_DL_Summer2026.git](https://github.com/wwnglv/Dazhy-Segbe_DL_Summer2026.git)
cd Dazhy-Segbe_DL_Summer2026
pip install -r requirements.txt
pip install torchaudio==2.2.0 --no-deps
```

## Training

```bash
python3 train.py --config-dir=src/configs --config-name=lcnn_asvspoof_longtrain

```

Config: `src/configs/lcnn_asvspoof_longtrain.yaml` (seed=42, 60 epochs, batch_size=32).

## Generating Predictions and Evaluation

```bash
python3 generate_predictions.py   # saves scores to students_solutions/
python3 grading.py                # calculates EER and score in grades.csv

```

## Project Structure

```
src/
├── model/lcnn.py         — LCNN architecture
├── datasets/asvspoof.py  — dataset + LFCC preprocessing
├── metrics/eer.py        — Equal Error Rate metric
├── loss/example.py       — CrossEntropyLoss
└── configs/               — experiment configurations (Hydra)

```

## References

* [ASVspoof 2019 Evaluation Plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf)
* Wang X., Yamagishi J. *A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection*, 2021 — [arXiv:2103.11326](https://arxiv.org/abs/2103.11326)
* Lavrentyeva G. et al. *STC Antispoofing Systems for the ASVspoof2019 Challenge*, 2019 — [arXiv:1904.05576](https://arxiv.org/abs/1904.05576)

```

```
