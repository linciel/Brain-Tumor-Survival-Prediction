"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM


# 原始的U-Net模型
class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, bins_num=6,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]


        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)
        self.gap = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.linear_p = nn.Linear(4, 16)
        self.linear_p2 = nn.Linear(16, 16)
        self.linear1 = nn.Linear(4 * 4 * 4 * 192,  4 * 4 * 128)

        self.linear2 = nn.Linear( 4 * 4 * 128,  4 * 128)
        self.linear3 = nn.Linear(4 * 128,  128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(80, 32)
        self.linear6 = nn.Linear(32, 1)

        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,x2):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        x2=self.linear_p(x2)
        x2=self.prelu(x2)
        x2=self.linear_p2(x2)
        x1=self.gap(bottom_2).flatten(1)
        x1 = self.linear1(x1)
        x1 = self.relu(self.linear2(x1))
        x1 = self.relu(self.linear3(x1))
        x1 = self.relu(self.linear4(x1))
        x1=torch.cat((x1,x2),dim=1)
        x1 = self.prelu(self.linear5(x1))

        x1 = self.prelu(self.linear6(x1))

        # sur_out=self.sigmoid(x1)
        sur_out=x1
        out = self.outconv(up1)
        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out,sur_out



class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


# if __name__ == '__main__':
#     unet = Unet(4, 3, 48)
#     inputs = torch.rand((1, 4, 128, 128, 128))
#     out = unet(inputs)
class Unet2(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet2"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, bins_num=6,
                 **kwargs):
        super(Unet2, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print("特征的通道维度列表，width是初始的通道维度，之后每下采样一次，扩大一倍" + str(features))

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)
        self.gap = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.linear_p = nn.Linear(4, 16)
        self.linear_p2 = nn.Linear(16, 16)
        self.linear1 = nn.Linear(4 * 4 * 4 * 192,  4 * 4 * 128)

        self.linear2 = nn.Linear( 4 * 4 * 128,  4 * 128)
        self.linear3 = nn.Linear(4 * 128,  128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(80, 32)
        self.linear6 = nn.Linear(32, 1)
        self.linear21 = nn.Linear(4 * 4 * 4 * 192,  4 * 4 * 128)

        self.linear2_2 = nn.Linear(4 * 4 * 128,  4 * 128)
        self.linear2_3 = nn.Linear(4 * 128,  128)
        self.linear2_4 = nn.Linear(128, 64)
        self.linear2_5 = nn.Linear(80, 32)
        self.linear2_6 = nn.Linear(32, 3)
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,x2):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        x2=self.linear_p(x2)
        x2=self.prelu(x2)
        x2=self.linear_p2(x2)
        x1=self.gap(bottom_2).flatten(1)
        x1 = self.linear1(x1)
        x3 = x1.clone()
        x1 = self.relu(self.linear2(x1))
        x1 = self.relu(self.linear3(x1))
        x1 = self.relu(self.linear4(x1))
        x1=torch.cat((x1,x2),dim=1)
        x1 = self.prelu(self.linear5(x1))

        x1 = self.prelu(self.linear6(x1))


        x3 = self.relu(self.linear2_2(x3))
        x3 = self.relu(self.linear2_3(x3))
        x3 = self.relu(self.linear2_4(x3))
        x3=torch.cat((x3,x2),dim=1)
        x3 = self.prelu(self.linear2_5(x3))

        x3 = self.prelu(self.linear2_6(x3))


        # sur_out=self.sigmoid(x1)
        sur_out=x1
        out = self.outconv(up1)
        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out,sur_out,x3



class Unet3(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet2"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, bins_num=6,
                 **kwargs):
        super(Unet2, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print("特征的通道维度列表，width是初始的通道维度，之后每下采样一次，扩大一倍" + str(features))

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)
        self.gap = nn.AdaptiveAvgPool3d((4, 4, 4))

        self.linear1 = nn.Linear(4 * 4 * 4 * 192,  4 * 4 * 128)
        self.linear2 = nn.Linear( 4 * 4 * 128,  4 * 128)
        self.linear3 = nn.Linear(4 * 128,  128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, 1)
        self.linear21 = nn.Linear(4 * 4 * 4 * 192,  4 * 4 * 128)

        self.linear2_2 = nn.Linear(4 * 4 * 128,  4 * 128)
        self.linear2_3 = nn.Linear(4 * 128,  128)
        self.linear2_4 = nn.Linear(128, 64)
        self.linear2_5 = nn.Linear(64, 32)
        self.linear2_6 = nn.Linear(32, 3)
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        x1=self.gap(bottom_2).flatten(1)
        x1 = self.linear1(x1)
        x3 = x1.clone()
        x1 = self.relu(self.linear2(x1))
        x1 = self.relu(self.linear3(x1))
        x1 = self.relu(self.linear4(x1))
        x1 = self.prelu(self.linear5(x1))

        x1 = self.prelu(self.linear6(x1))


        x3 = self.relu(self.linear2_2(x3))
        x3 = self.relu(self.linear2_3(x3))
        x3 = self.relu(self.linear2_4(x3))
        x3 = self.prelu(self.linear2_5(x3))

        x3 = self.prelu(self.linear2_6(x3))


        # sur_out=self.sigmoid(x1)
        sur_out=x1
        out = self.outconv(up1)
        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out,sur_out,x3

