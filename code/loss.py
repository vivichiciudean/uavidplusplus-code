import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation with ignore index support."""
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (N, C, H, W)
        targets: (N, H, W)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        dice_per_class = []
        for cls in range(num_classes):
            if self.ignore_index is not None:
                mask = (targets != self.ignore_index)
            else:
                mask = torch.ones_like(targets, dtype=torch.bool)

            target_cls = (targets == cls) & mask
            prob_cls = probs[:, cls, :, :] * mask.float()

            intersection = torch.sum(prob_cls * target_cls.float())
            union = torch.sum(prob_cls) + torch.sum(target_cls.float())
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        return 1.0 - torch.mean(torch.stack(dice_per_class))

def extract_boundaries(mask, kernel_size=3):
    boundaries = np.zeros_like(mask, dtype=np.uint8)
    for cls in np.unique(mask):
        if cls == 255:  # ignore index
            continue
        cls_mask = (mask == cls).astype(np.uint8)
        grad = cv2.morphologyEx(cls_mask, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)))
        boundaries[grad > 0] = cls
    return boundaries

class ClassBalancedBoundaryIoULoss(torch.nn.Module):
    def __init__(self, num_labels=11, beta=0.9999, ignore_index=255, kernel_size=3):
        super().__init__()
        self.num_labels = num_labels
        self.beta = beta
        self.ignore_index = ignore_index
        self.kernel_size = kernel_size

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) raw predictions
        targets: (B, H, W) ground truth
        """
        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        total_loss = 0.0
        total_weight = 0.0

        for b in range(logits.size(0)):
            pred_np = preds[b].detach().cpu().numpy().astype(np.uint8)
            tgt_np = targets[b].detach().cpu().numpy().astype(np.uint8)

            # extract boundary maps
            pred_b = extract_boundaries(pred_np, self.kernel_size)
            tgt_b  = extract_boundaries(tgt_np, self.kernel_size)

            for cls in range(self.num_labels):
                if cls == self.ignore_index:
                    continue

                pred_cls = (pred_b == cls)
                tgt_cls  = (tgt_b == cls)

                inter = np.logical_and(pred_cls, tgt_cls).sum()
                union = np.logical_or(pred_cls, tgt_cls).sum()

                if union == 0:
                    continue

                biou = inter / union

                # effective number weighting
                n_c = (tgt_cls.sum() + 1)  # +1 to avoid log(0)
                alpha_c = (1 - self.beta) / (1 - self.beta**n_c)

                total_loss += alpha_c * (1 - biou)
                total_weight += alpha_c

        return total_loss / (total_weight + 1e-6)

class JointLoss(nn.Module):
    """L = λCE * CE + λDice * Dice + λB * Boundary."""
    def __init__(self, class_weights=None, lambda_ce=1.0, lambda_dice=1.0, lambda_b=0.1, ignore_index=255, num_labels=11):
        super(JointLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="mean", ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.boundary_loss = ClassBalancedBoundaryIoULoss(ignore_index=ignore_index, num_labels=num_labels)
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_b = lambda_b

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        b_loss = self.boundary_loss(logits, targets)
        return self.lambda_ce * ce + self.lambda_dice * dice + self.lambda_b * b_loss
