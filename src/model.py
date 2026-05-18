"""
GoogLeNet (Inception V1) Architecture
======================================
Custom implementation of GoogLeNet for Blood Cell Classification.
~6.8M parameters with 9 Inception modules and 2 auxiliary classifiers.

Reference: Szegedy et al., "Going Deeper with Convolutions", CVPR 2015.

Architecture highlights:
  - Inception modules with parallel 1x1, 3x3, 5x5 convolutions + max pooling
  - 1x1 convolutions for dimensionality reduction
  - Two auxiliary classifiers for gradient injection during training
  - Global Average Pooling instead of FC layers
  - BatchNorm after every convolution for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Inception module with four parallel branches:
      Branch 1: 1x1 conv
      Branch 2: 1x1 conv -> 3x3 conv
      Branch 3: 1x1 conv -> 5x5 conv
      Branch 4: 3x3 max pool -> 1x1 conv
    All outputs are concatenated along the channel dimension.
    """

    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj, use_batchnorm=True):
        super(InceptionModule, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 reduce -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 reduce -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 max pool -> 1x1 projection
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier for gradient injection during training.
    Attached after inception_4a and inception_4d.

    Structure:
      AvgPool(5x5, stride=3) -> Conv 1x1(128) -> BN -> ReLU
      -> Flatten -> FC(1024) -> ReLU -> Dropout(0.7) -> FC(num_classes)
    """

    def __init__(self, in_channels, num_classes, dropout=0.7):
        super(AuxiliaryClassifier, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception V1) for Blood Cell Classification.

    Architecture (following original paper Table 1):
      Stem: Conv7x7 -> MaxPool -> Conv1x1 -> Conv3x3 -> MaxPool
      Inception 3a, 3b -> MaxPool
      Inception 4a, 4b, 4c, 4d, 4e -> MaxPool
      Inception 5a, 5b
      Global Average Pooling -> Dropout -> FC(num_classes)

    Two auxiliary classifiers are attached after inception_4a and inception_4d
    during training for gradient injection (weighted 0.3).

    Parameters: ~6.8M (with BatchNorm)
    """

    def __init__(self, num_classes=4, dropout=0.4, aux_dropout=0.7,
                 use_batchnorm=True):
        super(GoogLeNet, self).__init__()

        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm

        # ===== STEM =====
        self.stem = nn.Sequential(
            # Conv1: 3 -> 64, 7x7, stride 2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Conv2: 64 -> 64, 1x1 (dimensionality reduction)
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),

            # Conv3: 64 -> 192, 3x3
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ===== INCEPTION MODULES =====
        # Channel configs from the original paper (Table 1):
        #   (in_ch, 1x1, 3x3_red, 3x3, 5x5_red, 5x5, pool_proj)

        # Stage 3: Inception 3a, 3b
        self.inception_3a = InceptionModule(192,  64,  96, 128, 16, 32, 32, use_batchnorm)
        # Output: 64+128+32+32 = 256
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64, use_batchnorm)
        # Output: 128+192+96+64 = 480

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 4: Inception 4a, 4b, 4c, 4d, 4e
        self.inception_4a = InceptionModule(480, 192,  96, 208, 16, 48, 64, use_batchnorm)
        # Output: 192+208+48+64 = 512
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64, use_batchnorm)
        # Output: 160+224+64+64 = 512
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64, use_batchnorm)
        # Output: 128+256+64+64 = 512
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64, use_batchnorm)
        # Output: 112+288+64+64 = 528
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128, use_batchnorm)
        # Output: 256+320+128+128 = 832

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 5: Inception 5a, 5b
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128, use_batchnorm)
        # Output: 256+320+128+128 = 832
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128, use_batchnorm)
        # Output: 384+384+128+128 = 1024

        # ===== AUXILIARY CLASSIFIERS =====
        self.aux1 = AuxiliaryClassifier(512, num_classes, aux_dropout)   # after 4a
        self.aux2 = AuxiliaryClassifier(528, num_classes, aux_dropout)   # after 4d

        # ===== CLASSIFIER =====
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 3
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool3(x)

        # Stage 4
        x = self.inception_4a(x)

        # Auxiliary classifier 1 (after 4a)
        aux1_out = None
        if self.training:
            aux1_out = self.aux1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)

        # Auxiliary classifier 2 (after 4d)
        aux2_out = None
        if self.training:
            aux2_out = self.aux2(x)

        x = self.inception_4e(x)
        x = self.maxpool4(x)

        # Stage 5
        x = self.inception_5a(x)
        x = self.inception_5b(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training:
            return x, aux1_out, aux2_out
        return x

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def get_model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model summary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"  GoogLeNet (Inception V1) - Blood Cell Classifier")
    print("=" * 60)
    print(f"  Classes:            {model.num_classes}")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Model size:         {total_params * 4 / 1e6:.2f} MB (float32)")
    print("-" * 60)

    # Test forward pass
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size).to(device)

    model.train()
    out_train = model(dummy)
    print(f"  Train output:       main={out_train[0].shape}, "
          f"aux1={out_train[1].shape}, aux2={out_train[2].shape}")

    model.eval()
    with torch.no_grad():
        out_eval = model(dummy)
    print(f"  Eval output:        {out_eval.shape}")
    print("=" * 60)

    return total_params


if __name__ == "__main__":
    model = GoogLeNet(num_classes=4)
    get_model_summary(model)
