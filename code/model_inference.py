import os
import sys

import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torch.nn.functional as F

import albumentations as A
import torch

from dataloader import SegmentationDataset, rgb2id, num_labels, id2color_np, id2name
from inference_strategy import inference_tiled
from dinov3_wrapper import DinoV3SemanticSegmentationRegisters
from metrics import Evaluator

# ---------------- config (!must match training checkpoint!) ----------------
WEIGHTS_NAME = "./../models/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
# WEIGHTS_NAME = "./../models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
# WEIGHTS_NAME = "../models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

MODEL_NAME = "dinov3_vit7b16"
# MODEL_NAME = "dinov3_vith16plus"
# MODEL_NAME = "dinov3_vitl16"
# MODEL_NAME = "dinov3_vitb16"
# MODEL_NAME = "dinov3_vits16plus"
# MODEL_NAME = "dinov3_vits16"

REPO_NAME = '../models/dinov3'

root_dir='../data'
#modify this if you want to test on UDD/VDD/OpenEarthMap
test_folder = 'test'

version = "dinov3_vith+_unet_infererence_uavid"
output_dir = "../output/test_predictions_" + version
os.makedirs(output_dir, exist_ok=True)
CHECKPOINT_PATH = "../output/best_model.pth"

DINO_SIZE = 16
HEIGHT = 1088 * 1
WIDTH = 1088 * 1
IGNORE_INDEX = 255

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

test_transform = A.Compose([
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
])


test_dataset = SegmentationDataset(root_dir=root_dir, data_subset=test_folder,
                                   transform=test_transform, rgb2id=rgb2id,
                                   ignore_index=IGNORE_INDEX)

print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DinoV3SemanticSegmentationRegisters(
    num_labels=num_labels,
    repo_name=REPO_NAME,
    model_name=MODEL_NAME,
    half_precision=False,
    device=device,
    class_weights=None,
    weights_name=WEIGHTS_NAME,
    patch_size=DINO_SIZE,
    ignore_index=IGNORE_INDEX,
)
model.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded {CHECKPOINT_PATH} (epoch {checkpoint.get('epoch', '?')}, "
      f"best val IoU {checkpoint.get('best_val_iou', float('nan')):.4f})")

model.eval()
metric_test = Evaluator(num_classes=num_labels, ignore_index=IGNORE_INDEX)

with torch.no_grad():
    for idx in tqdm(range(len(test_dataset)), desc="Tiled test inference"):
        image, label, mask_name = test_dataset[idx]
        image = image.to(device)

        if "Earth" in test_folder: #resize to 4k for OpenEarthMap, since the images are low resolution and full of small details
            orig_h, orig_w = image.shape[-2:]
            image = F.interpolate(image.unsqueeze(0), scale_factor=4,
                                  mode="bicubic", align_corners=False).squeeze(0)

        logits_full = inference_tiled(model, image, tile_size=(HEIGHT, WIDTH), overlap=64, num_labels=num_labels, device=device)


        if "Earth" in test_folder:
            logits_full = F.interpolate(logits_full, size=(orig_h, orig_w),
                                        mode="bilinear", align_corners=False)

        pred = logits_full.argmax(dim=1).cpu().numpy()[0]

        if "Earth" in test_folder:
            label_np = label.numpy()
            pred[pred == 1] = 7  # building (wall) -> roof
            pred[pred == 6] = 0  # NO static car, dynamic car, sky, and human
            pred[pred == 8] = 0
            pred[pred == 9] = 0
            pred[pred == 10] = 0
            label = torch.from_numpy(label_np)

        if "claravid" in test_folder:
            label_np = label.numpy()
            pred[pred == 7] = 1
            pred[pred == 8] = 9
            pred[pred == 6] = 0
            label = torch.from_numpy(label_np)

        if "udd" in test_folder or "vdd" in test_folder:
            label_np = label.numpy()
            pred[pred == 4] = 3
            pred[pred == 8] = 9
            pred[pred == 6] = 0
            pred[pred == 10] = 0
            label = torch.from_numpy(label_np)

        metric_test.add_batch(gt_image=label.numpy(), pre_image=pred)

        color_mask = id2color_np[pred]
        out_path = os.path.join(output_dir, mask_name)
        Image.fromarray(color_mask).save(out_path)

per_class_iou_test = metric_test.Intersection_over_Union()
mean_iou_test = np.nanmean(per_class_iou_test)
overall_accuracy_test = metric_test.OA()
print("Test mIoU:", mean_iou_test)
print("Test OA:", overall_accuracy_test)
print("Per-class IoU:")
for cls_id, iou in enumerate(per_class_iou_test):
    print(f"Class {id2name[cls_id]}: {iou:.4f}")
