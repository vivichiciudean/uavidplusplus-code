import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from utils import *
from dataloader import *
from dinov3_wrapper import *
from inference_strategy import *
from metrics import *

import sys
import datetime
import gc

import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
import torchvision
from albumentations import Compose, Normalize
import random

SEED = 42  # choose any integer
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# WEIGHTS_NAME = "./../models/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
# WEIGHTS_NAME = "./../models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
WEIGHTS_NAME = "../models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# MODEL_NAME = "dinov3_vitl16"
# MODEL_NAME = "dinov3_vit7b16"
# MODEL_NAME = "dinov3_vit7b16"
# MODEL_NAME = "dinov3_vith16plus"
# MODEL_NAME = "dinov3_vitl16"
# MODEL_NAME = "dinov3_vitb16"
# MODEL_NAME = "dinov3_vits16plus"
MODEL_NAME = "dinov3_vits16"

root_dir= '../data'
REPO_NAME = '../models/dinov3'

learning_rate = 3e-5
weight_decay = 0.001
epochs = 40
version = "9_dinov3_vits_unet_inv_uavid++"

eval_only = False

resume = False
resume_checkpoint_name = "./../output/best_model.pth"

SAVE_EACH_N_EPOCH = 50
save_last = False

HEIGHT = 1088*1
WIDTH = 1088*1
DINO_SIZE = 16
TOKEN_W = WIDTH // DINO_SIZE
TOKEN_H = HEIGHT // DINO_SIZE
BATCH_SIZE = 8
start_epoch = 0
IGNORE_INDEX = 255
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

os.makedirs("../output", exist_ok=True)
log_file = f"../output/training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = Logger(log_file)

print("lr")
print(learning_rate)
print("epochs")
print(epochs)
print("version")
print(version)
print("W,H")
print(WIDTH, HEIGHT)
print("Batch size")
print(BATCH_SIZE)
print("Weight decay")
print(weight_decay)

print(num_labels)
print("ID to RGB mapping:", id2rgb)
print("RGB to ID mapping:", rgb2id)
print("ID to color mapping:", id2color_np)
print(id2name)

train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=[-0.1, 0.1], scale_limit=0.1, rotate_limit=[-10, 10],
                       mask_interpolation=cv2.INTER_NEAREST, rotate_method="ellipse", fill=0, fill_mask=IGNORE_INDEX, p=0.5),
    A.HorizontalFlip(p=0.5),

    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.25),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=3, p=0.25),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.25),
    A.RandomGamma(gamma_limit=(80, 120), p=0.25),

    A.CoarseDropout(num_holes_range=[1, 16], hole_height_range=[16, 128], hole_width_range=[16, 128], fill=0,
                    fill_mask=IGNORE_INDEX, p=0.5),

    A.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = A.Compose([
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
])


test_transform = A.Compose([
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
])


print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"


train_dataset = SegmentationDataset( root_dir=root_dir, data_subset='train',transform=train_transform, rgb2id=rgb2id, ignore_index=IGNORE_INDEX)
if eval_only:
    train_class_weights = None
else:
    train_class_weights = compute_class_weights(train_dataset, num_labels=num_labels, ignore_index=IGNORE_INDEX, important_classes=None).to(device)
val_dataset = SegmentationDataset(root_dir=root_dir,data_subset='val', transform=val_transform, rgb2id=rgb2id, ignore_index=IGNORE_INDEX)
test_dataset = SegmentationDataset(root_dir=root_dir,data_subset='test', transform=test_transform, rgb2id=rgb2id, ignore_index=IGNORE_INDEX)

print("train dataset size:", train_dataset.__len__())
print("train class weights:", train_class_weights)
print("val dataset size:",val_dataset.__len__())
print("test dataset size:",test_dataset.__len__())

if not eval_only:
    # get an item from the dataset, quick test everything is ok
    image, mask, _ = train_dataset[10]
    image = image.permute(1, 2, 0).numpy()

    mask_np = mask.detach().cpu().numpy()
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    # convert id to rgb values in mask
    for i in range(mask_np.shape[0]):
        for j in range(mask_np.shape[1]):
            pixel_value = int(mask_np[i, j])
            if pixel_value == 255:
                color_mask[i, j, :] = [255, 255, 255]
                continue
            color_mask[i, j, :] = id2color_np[pixel_value]
    color_mask = color_mask.astype(np.float32)
    color_mask /= 255.0

    # visualize the image and mask
    plt.figure()
    plt.imshow(mask)
    plt.savefig("../output/1_" + version + "_mask.png")
    # plt.show()

    plt.figure()
    plt.imshow(color_mask)
    plt.savefig("../output/2_"+ version+"_colormask.png")
    # plt.show()

    plt.figure()
    denorm_img = denormalize(image, imagenet_std, imagenet_mean)
    plt.imshow(denorm_img)
    plt.savefig("../output/3_"+ version+"_img.png")
    # plt.show()

    # Scale to [0,255] and convert to uint8
    img_vis = (denorm_img * 255).astype(np.uint8)

    # Prepare mask in [0,255] and 3-channel
    mask_vis = (color_mask*255).astype(np.uint8)  # keep RGB

    # fuse mask and image into a single image
    fused_img = cv2.addWeighted(img_vis, .5, mask_vis, 0.5, 0)
    plt.figure()
    plt.imshow(fused_img)
    plt.savefig("../output/4_"+ version+"_overlayed.png")
    # plt.show()

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)

# load a sample batch of data to make sure eveything is working
batch = next(iter(train_dataloader))

for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
print(batch["pixel_values"].dtype)
print(batch["labels"].dtype)

# intialize model
model = DinoV3SemanticSegmentationRegisters(
    num_labels=num_labels,
    repo_name=REPO_NAME,
    model_name=MODEL_NAME,
    half_precision=False,
    device=device,
    class_weights=train_class_weights,
    weights_name=WEIGHTS_NAME,
    patch_size=DINO_SIZE,
    image_w = WIDTH,
    image_h = HEIGHT,
    ignore_index=IGNORE_INDEX
)
model.to(device)

# set optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

warmup_epochs = 5

num_training_steps = epochs * len(train_dataloader)
warmup_iters = warmup_epochs * len(train_dataloader)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_training_steps - warmup_iters, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters]
)

best_val_iou = 0.0
if resume == True:
    print("Resume training from checkpoint ", resume_checkpoint_name)
    checkpoint = torch.load(resume_checkpoint_name, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_iou = checkpoint.get("best_val_iou", 0.0)

    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

filename = "./model_" + version + "_"

# Initialize for train, val, test
metric_train = Evaluator(num_classes=num_labels, ignore_index=IGNORE_INDEX)
metric_val = Evaluator(num_classes=num_labels, ignore_index=IGNORE_INDEX)
metric_test = Evaluator(num_classes=num_labels, ignore_index=IGNORE_INDEX)


history_classwise_iou_train = []
history_classwise_iou_val = []

history_loss_train = []
history_loss_val = []
history_mean_iou_train = []
history_mean_iou_val = []
history_mean_accuracy_train = []
history_mean_accuracy_val = []

if not eval_only:
    # start training
    for epoch in range(start_epoch, epochs):
        if epoch % SAVE_EACH_N_EPOCH == 0 and epoch != 0:
            pth_filename = filename + "epoch_" + str(epoch) + ".pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, pth_filename)

        running_loss = 0.0
        n_samples = 0
        print("Epoch:", epoch + 1)

        model.classifier.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            running_loss += loss.item() * pixel_values.size(0)
            n_samples += pixel_values.size(0)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()

            predicted = outputs.logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            metric_train.add_batch(gt_image=labels_np, pre_image=predicted)

        per_class_iou_train = metric_train.Intersection_over_Union()
        mean_iou_train = np.nanmean(per_class_iou_train)
        overall_accuracy_train = metric_train.OA()

        avg_train_loss = running_loss / n_samples

        history_loss_train.append(avg_train_loss)
        history_mean_iou_train.append(mean_iou_train)
        history_mean_accuracy_train.append(overall_accuracy_train)
        history_classwise_iou_train.append(per_class_iou_train)

        running_val_loss = 0.0
        n_samples_val = 0
        model.eval()

        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values, labels=labels)
                val_loss = outputs.loss

                running_val_loss += val_loss.item() * pixel_values.size(0)
                n_samples_val += pixel_values.size(0)

                predicted = outputs.logits.argmax(dim=1).detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                metric_val.add_batch(gt_image=labels_np, pre_image=predicted)

        per_class_iou_val = metric_val.Intersection_over_Union()
        mean_iou_val = np.nanmean(per_class_iou_val)
        overall_accuracy_val = metric_val.OA()

        avg_val_loss = running_val_loss / n_samples_val
        history_loss_val.append(avg_val_loss)

        history_mean_iou_val.append(mean_iou_val)
        history_mean_accuracy_val.append(overall_accuracy_val)
        history_classwise_iou_val.append(per_class_iou_val)

        print("Train Loss:", avg_train_loss, "Val Loss:", avg_val_loss)
        print("Train mIoU:", mean_iou_train, "Val mIoU:", mean_iou_val)
        print("Train OA:", overall_accuracy_train, "Val OA:", overall_accuracy_val)

        print("Per-class train IoU:")
        for idx, iou in enumerate(per_class_iou_train):
            print(f"  {id2name[idx]}: {iou:.4f}")

        print("Per-class val IoU:")
        for idx, iou in enumerate(per_class_iou_val):
            print(f"  {id2name[idx]}: {iou:.4f}")
        # Save best model
        if mean_iou_val > best_val_iou:
            best_val_iou = mean_iou_val

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou
            }
            torch.save(checkpoint, "../output/best_model.pth")

            print(f"✅ New best model saved at epoch {epoch+1} with IoU {best_val_iou:.4f}")

        metric_train.reset()
        metric_val.reset()

    plt.figure()
    plt.plot(history_loss_train, label="train_loss")
    plt.plot(history_loss_val, label="val_loss")
    plt.legend()
    plt.savefig("../output/5_"+ version+"_loss.png")
    # plt.show()

    # # plot the training and validation mean iou
    plt.figure()
    plt.plot(history_mean_iou_train, label="train_mean_iou")
    plt.plot(history_mean_iou_val, label="val_mean_iou")
    plt.legend()
    plt.savefig("../output/6_"+ version+"_iou.png")
    # plt.show()

    # plot the training and validation mean accuracy
    plt.figure()
    plt.plot(history_mean_accuracy_train, label="train_mean_accuracy")
    plt.plot(history_mean_accuracy_val, label="val_mean_accuracy")
    plt.legend()
    plt.savefig("../output/7_"+ version+"_accuracy.png")
    # plt.show()

    if save_last:
        pth_filename = filename + "epoch_" + str(epochs) + "_final.pth"
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, pth_filename)


# Load the BEST checkpoint
pth_filename = "../output/best_model.pth"
checkpoint = torch.load(pth_filename, weights_only=False)

# Load the saved states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']



# --- Directory to save test predictions ---
output_dir = "../output/test_predictions_" + version
os.makedirs(output_dir, exist_ok=True)

model.eval()
# --- Loop over test dataset ---
with torch.no_grad():
    for idx in tqdm(range(len(test_dataset)), desc="Tiled test inference"):
        image, label, mask_name = test_dataset[idx]
        image = image.to(device)

        logits_full = inference_tiled(model, image, tile_size=(HEIGHT, WIDTH), overlap=64, num_labels=num_labels, device=device)
        pred = logits_full.argmax(dim=1).cpu().numpy()[0]

        metric_test.add_batch(gt_image=label.numpy(), pre_image=pred)

        # save prediction as color mask
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
