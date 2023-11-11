import torch
import copy
import torch.nn.functional as F
from torchvision import models

class SaliconNet(torch.nn.Module):

    def resnet_backbone(self):
        backbone = models.resnet34()
        children = list(backbone.children())[:-2]
        backbone = torch.nn.Sequential(*children)
        return backbone

    def __init__(self):
        super(SaliconNet, self).__init__()

        backbone = self.resnet_backbone()
        self.coarse_backbone = backbone
        self.fine_backbone = copy.deepcopy(backbone)
        self.conv1x1 = torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

    def soft_binarization(self, x):
        x = torch.pow(x, 2)
        return x / (x + 0.1**2)

    def forward(self, x):

        _, _, h, w = x.shape

        fine = x
        coarse = F.interpolate(x, size=(int(h/2), int(w/2)), mode="bilinear")

        fine = self.fine_backbone(fine)
        _, _, fine_h, fine_w = fine.shape

        coarse = self.coarse_backbone(coarse)
        coarse = F.interpolate(coarse, size=(fine_h, fine_w), mode="bilinear")

        x = torch.cat((fine, coarse), dim=1)        
        x = self.conv1x1(x)

        x = self.sigmoid(x)
        x = self.soft_binarization(x)

        return x


class QuantizableSaliconNet(SaliconNet):

    def resnet_backbone(self):
        """
        Override this method to return a quant version resnet backbone.
        """
        backbone = models.quantization.resnet._resnet("resnet34", models.quantization.resnet.QuantizableBasicBlock, [3, 4, 6, 3], False, False, None)
        children = list(backbone.children())[:-4]
        backbone = torch.nn.Sequential(*children)
        return backbone

    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = super(QuantizableSaliconNet, self).forward(x)
        x = self.dequant(x)
        return x
    
    def __find_conv_bn_relu(self, module: torch.nn.Module):
        list = []

        if isinstance(module, torch.nn.Sequential):
            for i, module_i in enumerate(module):
                names = self.__find_conv_bn_relu(module_i)
                list.extend([f"{i}.{name}" for name in names])
        if isinstance(module, models.resnet.BasicBlock):
            return ['conv1', 'bn1', 'relu']
        
        return list

    def __find_conv_bn(self, module: torch.nn.Module):
        list = []

        if isinstance(module, torch.nn.Sequential):
            for i, module_i in enumerate(module):
                names = self.__find_conv_bn(module_i)
                list.extend([f"{i}.{name}" for name in names])
        if isinstance(module, models.resnet.BasicBlock):
            return ['conv2', 'bn2']
        
        return list

    def __fuse_backbone(self, backbone):
        fuse_conv_bn_relu = self.__find_conv_bn_relu(backbone)
        fuse_conv_bn = self.__find_conv_bn(backbone)

        list = []
        for i in range(0, int(len(fuse_conv_bn_relu) / 3)):
            j = 3 * i
            list.append([fuse_conv_bn_relu[j], fuse_conv_bn_relu[j+1], fuse_conv_bn_relu[j+2]])

        for i in range(0, int(len(fuse_conv_bn) / 2)):
            j = 2 * i
            list.append([fuse_conv_bn[j], fuse_conv_bn[j+1]])
        
        return torch.quantization.fuse_modules(backbone, list, inplace=True)

    def fuse_model(self):
        self.__fuse_backbone(self.fine_backbone)
        self.__fuse_backbone(self.coarse_backbone)
        