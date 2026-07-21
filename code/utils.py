import torch
import torch.nn as nn
from tqdm.auto import tqdm
from collections import Counter, OrderedDict
import numpy as np
import sys
import random

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    return batch


def seed_worker(worker_id, SEED):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_class_weights(dataset, num_labels, ignore_index=255, important_classes=None, boost=2):
    counts = Counter()
    total = 0

    for _, mask, _ in tqdm(dataset, desc="Computing class frequencies"):
        mask_np = mask.numpy().flatten()
        mask_np = mask_np[mask_np != ignore_index]  # ignore background if needed
        counts.update(mask_np.tolist())
        total += mask_np.size

    # Normalize counts to frequencies
    class_freq = np.array([counts[i] if i in counts else 0 for i in range(num_labels)], dtype=np.float32)
    print(class_freq)
    class_freq = class_freq / class_freq.sum()


    # Inverse frequency weighting (or sqrt-inverse to soften extreme imbalance)
    weights = 1.0 / np.sqrt(class_freq + 1e-6)
    # Apply extra boost only to explicitly defined important classes
    if important_classes is not None:
        for cid in important_classes:
            if cid < num_labels:
                weights[cid] *= boost
            else:
                print("Something is wrong with your class weights")
    weights = weights / weights.sum() * num_labels  # normalize for stability


    return torch.tensor(weights, dtype=torch.float32)

def denormalize(img_tensor, std, mean):
    """
    img_tensor: torch.Tensor (C,H,W) or (H,W,C), normalized
    returns: numpy array (H,W,C) in [0,1] for matplotlib
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img = img_tensor

    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img