import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

class LinearHead(torch.nn.Module):
    def __init__(self, in_channels, W=32, H=32, num_labels=1):
        super(LinearHead, self).__init__()

        self.in_channels = in_channels  # patch descriptor size
        self.width = W
        self.height = H

        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))


    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0, 3, 1, 2)
        return self.classifier(x)



class ConvHead(torch.nn.Module):
    def __init__(self, in_channels, W=32, H=32, num_labels=1):
        super(ConvHead, self).__init__()

        self.in_channels = in_channels  # patch descriptor size
        self.width = W
        self.height = H

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_channels // 2),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(in_channels // 2, num_labels, kernel_size=1)
        )

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0, 3, 1, 2)
        return self.classifier(x)




class BlockDown(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.block(x)
        down = self.pool(out)
        return down, out  # downsampled + skip connection


class BlockUp(nn.Module):
    """Up-conv -> concat skip -> Conv -> BN -> ReLU"""
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        # copy & crop if needed
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetHead(nn.Module):
    # The down path progressively reduces the number of channels, while the up
    # path increases them. Since the DINO output is already a high-dimensional
    # feature embedding, the network first compresses the representation before
    # expanding it during decoding.
    def __init__(self, in_channels, num_labels, H, W):
        super().__init__()
        self.in_channels = in_channels
        self.height = H
        self.width = W

        # Down blocks
        self.block_down1 = BlockDown(in_channels, in_channels // 2)
        self.block_down2 = BlockDown(in_channels // 2, in_channels // 4)
        self.block_down3 = BlockDown(in_channels // 4, in_channels // 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=True)
        )

        # Up blocks
        self.block_up3 = BlockUp(in_channels // 16, in_channels // 8, in_channels // 8)
        self.block_up2 = BlockUp(in_channels // 8, in_channels // 4, in_channels // 4)

        # There is no need to increase the number of channels to `in_channels`, as it
        # would only double the number of parameters without improving performance.
        self.block_up1 = BlockUp(in_channels // 4, in_channels // 2, in_channels // 2)


        # Final 1x1 conv
        self.final_conv = nn.Conv2d(in_channels // 2, num_labels, kernel_size=1)

    def forward(self, embeddings):
        # print(embeddings.shape)
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        x = embeddings.permute(0, 3, 1, 2)
        # print(x.shape)

        d1, skip1 = self.block_down1(x)
        d2, skip2 = self.block_down2(d1)
        d3, skip3 = self.block_down3(d2)

        bottleneck = self.bottleneck(d3)

        u3 = self.block_up3(bottleneck, skip3)
        u2 = self.block_up2(u3, skip2)
        u1 = self.block_up1(u2, skip1)

        logits = self.final_conv(u1)
        return logits
