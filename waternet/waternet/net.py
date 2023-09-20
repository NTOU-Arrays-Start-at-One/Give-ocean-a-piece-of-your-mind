import torch
import torch.nn as nn
from waternet.resnet import resnet50

# import torch.nn.functional as F


class upSample(nn.Module):
    def __init__(self, in_size, out_size):
        super(upSample, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class ConfidenceMapGenerator(nn.Module):
    def __init__(self, pretrained=False, backbone="resnet50", num_classes=3):
        super().__init__()
        # Confidence maps
        # Accepts input of size (N, 3*4, H, W)
        self.resnet = resnet50(pretrained=pretrained)
        in_filters = [192, 512, 1536, 3072]
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = upSample(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = upSample(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = upSample(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = upSample(in_filters[0], out_filters[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0],
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0],
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.final_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.backbone = backbone
        # self.conv1 = nn.Conv2d(
        #     in_channels=12, out_channels=128, kernel_size=7, dilation=1, padding="same"
        # )
        # self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(
        #     in_channels=128, out_channels=128, kernel_size=5, dilation=1, padding="same"
        # )
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(
        #     in_channels=128, out_channels=128, kernel_size=3, dilation=1, padding="same"
        # )
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(
        #     in_channels=128, out_channels=64, kernel_size=1, dilation=1, padding="same"
        # )
        # self.relu4 = nn.ReLU()
        # self.conv5 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=7, dilation=1, padding="same"
        # )
        # self.relu5 = nn.ReLU()
        # self.conv6 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=5, dilation=1, padding="same"
        # )
        # self.relu6 = nn.ReLU()
        # self.conv7 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding="same"
        # )
        # self.relu7 = nn.ReLU()
        # self.conv8 = nn.Conv2d(
        #     in_channels=64, out_channels=3, kernel_size=3, dilation=1, padding="same"
        # )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)

        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(out)
        
        '''
        print("feat1", feat1.shape)
        print("feat2", feat2.shape)
        print("feat3", feat3.shape)
        print("feat4", feat4.shape)
        print("feat5", feat5.shape)
        '''

        up3 = self.up_concat3(feat3, feat4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        final = self.final(up1)
        final = self.final_up(final)
        # out = torch.cat([x, wb, ce, gc], dim=1)
        # 編碼器部分

        # out = self.relu1(self.conv1(out))
        # out = self.relu2(self.conv2(out))
        # out = self.relu3(self.conv3(out))
        # out = self.relu4(self.conv4(out))
        # out = self.relu5(self.conv5(out))
        # out = self.relu6(self.conv6(out))
        # out = self.relu7(self.conv7(out))
        # out = self.sigmoid(self.conv8(out))
        # print("final", final.shape)
        out1, out2, out3 = torch.split(final, [1, 1, 1], dim=1)
        return out1, out2, out3

    def freeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.resnet.parameters():
            param.requires_grad = True


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=7, dilation=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, dilation=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, xbar):
        out = torch.cat([x, xbar], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class WaterNet(nn.Module):
    """
    waternet = WaterNet()
    in = torch.randn(16, 3, 112, 112)
    waternet_out = waternet(in, in, in, in)
    waternet_out.shape
    # torch.Size([16, 3, 112, 112])
    """

    def __init__(self):
        super().__init__()
        self.cmg = ConfidenceMapGenerator()
        self.wb_refiner = Refiner()
        self.ce_refiner = Refiner()
        self.gc_refiner = Refiner()

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc)

        '''
        print("wb_cm", wb_cm.shape)
        print("ce_cm", ce_cm.shape)
        print("gc_cm", gc_cm.shape)
        '''

        refined_wb = self.wb_refiner(x, wb)
        refined_ce = self.ce_refiner(x, ce)
        refined_gc = self.gc_refiner(x, gc)
        return (
            torch.mul(refined_wb, wb_cm)
            + torch.mul(refined_ce, ce_cm)
            + torch.mul(refined_gc, gc_cm)
        )
