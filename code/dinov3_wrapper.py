import torch
from transformers.modeling_outputs import SemanticSegmenterOutput
from segmentation_heads import *
from loss import *

class DinoV3SemanticSegmentationRegisters(torch.nn.Module):
    def __init__(self, num_labels=1, repo_name="facebookresearch/dinov3", model_name="dinov3_vitl16",
                 half_precision=False, device="cuda", class_weights=None, weights_name="sat.pth", patch_size=16,
                 image_w=1088, image_h=1088, ignore_index=255):
        super().__init__()
        self.class_weights = class_weights
        self.repo_name = repo_name
        self.model_name = model_name
        self.device = device
        self.half_precision = half_precision
        self.patch_size = patch_size
        self.num_labels = num_labels
        self.weights_name = weights_name
        self.ignore_index = ignore_index
        self.image_w = image_w
        self.image_h = image_h
        # self.anyup = torch.hub.load('wimmerth/anyup', 'anyup')
        # self.anyup = torch.hub.load('wimmerth/anyup', 'anyup_multi_backbone', use_natten=True)

        try:
            if self.half_precision:
                print("Carefull, loading half precision!")
                self.backbone = torch.hub.load(repo_or_dir=self.repo_name, model=self.model_name).half().to(self.device)
            else:
                self.backbone = torch.hub.load(repo_or_dir=self.repo_name, model=self.model_name, source='local', weights=self.weights_name)
                # self.backbone = torch.hub.load(repo_or_dir=self.repo_name, model=self.model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv3 backbone '{self.model_name}' from '{self.repo_name}' via torch.hub. "
                "Check that the repo is available locally or that you have the correct weight URLs/permissions. "
                "See the DINOv3 README for instructions (torch.hub load examples)."
            ) from e

        # Put backbone in eval/frozen mode by default, since we only train the head.
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        print("In channels dim dinov3 (should be 4096 for vit7b)")
        in_channels = int(self.backbone.embed_dim)
        print(in_channels)

        self.target_channels =  in_channels
        self.scale_factor = 1

        H = self.image_h // self.patch_size
        W = self.image_w // self.patch_size

        # Modify here if you want to use a different Head (Linear/Conv/UNet-inspired)
        self.classifier = UNetHead(self.target_channels, num_labels=self.num_labels, H=H, W=W).to(self.device)

        trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    def forward(self, pixel_values, labels=None):
        # pixel_values: [B, 3, H, W]
        pixel_values = pixel_values.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        with torch.no_grad():
            feat = self.backbone.get_intermediate_layers(pixel_values, n=1, reshape=True)[0]  # [B, C, h, w]
        patch_embeddings = feat.permute(0, 2, 3, 1).contiguous()  # [B, h, w, C] for the head

        logits = self.classifier(patch_embeddings)

        # upsample to original image resolution
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            criterion = JointLoss(class_weights=self.class_weights, lambda_ce=1, lambda_dice=1, lambda_b=1, ignore_index=self.ignore_index , num_labels=self.num_labels)
            loss = criterion(logits, labels)


        return SemanticSegmenterOutput(loss=loss, logits=logits)