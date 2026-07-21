import os
import sys
import numpy as np
from PIL import Image
from os.path import join, splitext
import torch
from torch.utils.data import Dataset

Color_Information = {
    "clutter": {"red_value": 0, "green_value": 0, "blue_value": 0, "class_value": 0},
    "wall": {"blue_value": 0, "green_value": 0, "red_value": 128, "class_value": 1},
    "road": {"blue_value": 128, "green_value": 64, "red_value": 128, "class_value": 2},
    "tree": {"blue_value": 0, "green_value": 128, "red_value": 0, "class_value": 3},
    "lowveg": {"blue_value": 0, "green_value": 128, "red_value": 128, "class_value": 4},

    "water": {"blue_value": 255, "green_value": 0, "red_value": 0, "class_value": 5},
    "sky": {"blue_value": 255, "green_value": 255, "red_value": 128, "class_value": 6},
    "roof": {"blue_value": 70, "green_value": 70, "red_value": 70, "class_value": 7},

    "staticcar": {"blue_value": 192, "green_value": 0, "red_value": 192, "class_value": 8},
    "dynamiccar": {"blue_value": 128, "green_value": 0, "red_value": 64, "class_value": 9},
    "human": {"blue_value": 0, "green_value": 64, "red_value": 64, "class_value": 10},
}


rgb2id = {(info["red_value"], info["green_value"], info["blue_value"]): info["class_value"] for info in Color_Information.values()}
id2rgb = {info["class_value"]:(info["red_value"], info["green_value"], info["blue_value"]) for info in Color_Information.values()}


num_labels = max(id2rgb.keys()) + 1
id2color_np = np.zeros((num_labels, 3), dtype=np.uint8)
for cid, rgb in id2rgb.items():
    id2color_np[cid] = rgb

id2name = {v["class_value"]: k for k,v in Color_Information.items()}

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP")

class SegmentationDataset(Dataset):
    def __init__(self, data_subset, root_dir, rgb2id, transform=None, ignore_index=255):
        self.data_subset = data_subset
        self.transform = transform
        self.root = root_dir
        self.ignore_index = ignore_index
        self.rgb2id = rgb2id

        self.img_dir = join(self.root, self.data_subset, "Images")
        self.label_dir = join(self.root, self.data_subset, "Labels")

        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f"Missing image directory: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise RuntimeError(f"Missing label directory: {self.label_dir}")

        # --- Collect masks (.png or .tif only) ---
        self.mask_files = sorted(
            [f for f in os.listdir(self.label_dir) if f.lower().endswith(".png") or f.lower().endswith(".tif")]
        )

        if len(self.mask_files) == 0:
            raise RuntimeError("No PNG mask files found.")

        self.image_map = {}
        for f in os.listdir(self.img_dir):
            stem, ext = splitext(f)
            if ext.lower() in IMG_EXTENSIONS:
                self.image_map.setdefault(stem, []).append(join(self.img_dir, f))

        self.samples = []
        for mask_file in self.mask_files:
            stem = splitext(mask_file)[0]

            if stem not in self.image_map:
                raise FileNotFoundError(f"No image found for mask: {mask_file}")

            if len(self.image_map[stem]) > 1:
                raise RuntimeError(
                    f"Multiple images found for '{stem}': {self.image_map[stem]}"
                )

            img_path = self.image_map[stem][0]
            mask_path = join(self.label_dir, mask_file)
            self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        target = Image.open(mask_path).convert("RGB")

        image = np.array(image)
        target = np.array(target)

        target_new = np.full(target.shape[:2], fill_value=self.ignore_index, dtype=np.int32)

        for (r, g, b), cls in self.rgb2id.items():
            mask = (
                (target[:, :, 0] == r) &
                (target[:, :, 1] == g) &
                (target[:, :, 2] == b)
            )
            target_new[mask] = cls

        if self.transform is not None:
            transformed = self.transform(image=image, mask=target_new)
            image = transformed["image"]
            target_new = transformed["mask"]

        image = torch.tensor(image).permute(2, 0, 1)
        target_new = torch.LongTensor(target_new)

        mask_name = os.path.basename(mask_path)
        return image, target_new, mask_name

