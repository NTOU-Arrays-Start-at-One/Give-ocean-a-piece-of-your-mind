import torch
import torch.nn as nn
from waternet.resnet import resnet50


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
        #Confidence maps
        #Accepts input of size (N, 3*4, H, W)
        self.resnet = resnet50(pretrained=pretrained)
        in_filters = [192, 512, 1536, 3072]
        out_filters = [64, 128, 256, 512]

        #upsampling
        #64,64,512
        self.up_concat4 = upSample(in_filters[3], out_filters[3])
        #128,128,256
        self.up_concat3 = upSample(in_filters[2], out_filters[2])
        #256,256,128
        self.up_concat2 = upSample(in_filters[1], out_filters[1])
        #512,512,64
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
