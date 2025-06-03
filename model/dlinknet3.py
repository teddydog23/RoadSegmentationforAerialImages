import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DBlock(nn.Module):
    def __init__(self, channels):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.dilate3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.dilate4 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        d1 = self.relu(self.dilate1(x))
        d2 = self.relu(self.dilate2(d1))
        d3 = self.relu(self.dilate3(d2))
        d4 = self.relu(self.dilate4(d3))

        return x + d1 + d2 + d3 + d4
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Simulate ExpandDims + Conv3DTranspose + Squeeze
        self.deconv3d = nn.ConvTranspose3d(
            mid_channels, mid_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            output_padding=(0, 1, 1)
        )

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Expand dims
        x = x.unsqueeze(2)  # (B, C, 1, H, W)
        x = self.deconv3d(x)
        x = x.squeeze(2)    # (B, C, H*2, W*2)

        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x

class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DLinkNet34, self).__init__()
        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=pretrained)

        # Encoder
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Center
        self.dblock = DBlock(512)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final conv
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, kernel_size=4, stride=2, padding=1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        center = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(center) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.sigmoid(out)
