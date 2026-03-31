import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class LightweightUNet(nn.Module):
    """
    U-Net architecture using a ResNet18 encoder trained from scratch. 
    Residual connections inherently solve the vanishing gradient / convergence plateaus.
    """
    def __init__(self, n_channels=3, n_classes=1, base_filters=32):
        super().__init__()
        self.n_classes = n_classes
        
        # ResNet18 Encoder (Weights=None trains it strictly from scratch)
        resnet = models.resnet18(weights=None)
        
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B, 64, H/2, W/2]
        self.down1 = nn.Sequential(resnet.maxpool, resnet.layer1)        # [B, 64, H/4, W/4]
        self.down2 = resnet.layer2                                       # [B, 128, H/8, W/8]
        self.down3 = resnet.layer3                                       # [B, 256, H/16, W/16]
        self.down4 = resnet.layer4                                       # [B, 512, H/32, W/32]
        
        # Bottleneck dropout is still highly effective
        self.bottleneck_drop = nn.Dropout2d(0.2)
        
        # Decoder (Merging deep features with skip connections)
        # ResNet features: x5=512, x4=256, x3=128, x2=64, x1=64
        self.up1 = Up(512 + 256, 256)    # 768 -> 256
        self.up2 = Up(256 + 128, 128)    # 384 -> 128
        self.up3 = Up(128 + 64, 64)      # 192 -> 64
        self.up4 = Up(64 + 64, 32)       # 128 -> 32
        
        # We need one final upsample to get back from H/2 to full H, since resnet.conv1 has stride=2
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Output layer
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.up1, self.up2, self.up3, self.up4]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.outc.weight, std=0.001)

    def forward(self, x):
        x1 = self.inc(x)        # 64 channels
        x2 = self.down1(x1)     # 64 channels
        x3 = self.down2(x2)     # 128 channels
        x4 = self.down3(x3)     # 256 channels
        x5 = self.down4(x4)     # 512 channels
        
        x5 = self.bottleneck_drop(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.up_final(x)
        logits = self.outc(x)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = LightweightUNet(n_classes=1)
    dummy_input = torch.randn(4, 3, 256, 512) 
    
    print(f"Testing ResNet18 U-Net Forward Pass...")
    out = model(dummy_input)
    
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {out.shape}   (Expected: [4, 1, 256, 512])")
    print(f"\nTotal Trainable Parameters: {count_parameters(model):,}")
