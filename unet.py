import torch

class DoubleConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBN, self).__init__()
        padding=1
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

class DeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConv, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class UNet(torch.nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.layer1 = DoubleConvBN(3, 64)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = DoubleConvBN(64, 128)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = DoubleConvBN(128, 256)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = DoubleConvBN(256, 512)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = DoubleConvBN(512, 1024)
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upsample1 = DeConv(1024, 512)
        self.layer6 = DoubleConvBN(1024, 512)
        self.upsample2 = DeConv(512, 256)
        self.layer7 = DoubleConvBN(512, 256)
        self.upsample3 = DeConv(256, 128)
        self.layer8 = DoubleConvBN(256, 128)
        self.upsample4 = DeConv(128, 64)
        self.layer9 = DoubleConvBN(128, 64)
        self.covn1x1 = torch.nn.Conv2d(in_channels = 64, out_channels=1, kernel_size = 1)

    # def _crop(self, x, x_):
    #     h_, w_ = x_.shape[2:]
    #     h, w = x.shape[2:]
    #     pad_h = int((h_ - h)/2)
    #     pad_w = int((w_ - w)/2)
    #     return x_[:, :, pad_h: h_-pad_h, pad_w: w_ - pad_w]

    def forward(self, x):

        x1 = self.layer1(x)
        x1_ = self.maxpool1(x1)

        x2 = self.layer2(x1_)
        x2_ = self.maxpool2(x2)

        x3 = self.layer3(x2_)
        x3_ = self.maxpool3(x3)

        x4 = self.layer4(x3_)
        x4_ = self.maxpool4(x4)
        
        x5 = self.layer5(x4_)
        x5 = self.upsample1(x5)

        # x4 = self._crop(x5, x4)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.layer6(x5)
        x6 = self.upsample2(x6)

        # x3 = self._crop(x6, x3)
        x6 = torch.cat([x3, x6], dim=1)

        x7 = self.layer7(x6)
        x7 = self.upsample3(x7)

        # x2 = self._crop(x7, x2)
        x7 = torch.cat([x2, x7], dim=1)

        x8 = self.layer8(x7)
        x8 = self.upsample4(x8)

        # x1 = self._crop(x8, x1)
        x8 = torch.cat([x1, x8], dim=1)

        x9 = self.layer9(x8)
        
        x10 = self.covn1x1(x9)

        x11 = torch.sigmoid(x10)
        return x11
