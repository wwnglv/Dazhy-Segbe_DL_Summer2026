import numpy as np
import torch
from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores)
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontar_trial_sums = np.cumsum(1 - labels)
    frr = 1 - tar_trial_sums / target_scores.size
    far = nontar_trial_sums / nontarget_scores.size
    return frr, far


def compute_eer(target_scores, nontarget_scores):
    frr, far = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer * 100.0


class ERRMetric(BaseMetric):
    def __init__(self, name="ERRMetric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.reset()

    def reset(self):
        self.target_scores = []
        self.nontarget_scores = []

    def update(self, logits, labels, **kwargs):
        probs = torch.softmax(logits, dim=-1)
        scores = probs[:, 0].detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()

        for score, lbl in zip(scores, lbls):
            if lbl == 1:
                self.target_scores.append(score)
            else:
                self.nontarget_scores.append(score)

    def __call__(self, logits, labels, **batch):
        self.update(logits, labels)
        return self.result()

    def result(self):
        if len(self.target_scores) == 0 or len(self.nontarget_scores) == 0:
            return 100.0
        return compute_eer(np.array(self.target_scores), np.array(self.nontarget_scores))
