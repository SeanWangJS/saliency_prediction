import torch
import copy
import torch.nn.functional as F
from torchvision import models

class SaliconNet(torch.nn.Module):

    def __resnet34_backbone(self):
        backbone = models.resnet34(pretrained=True)
        children = list(backbone.children())[:-2]
        backbone = torch.nn.Sequential(*children)
        return backbone

    def __vgg16_backbone(self):
        backbone = models.vgg16(pretrained=True)
        backbone = backbone.features[:30]        
        return backbone

    def __init__(self):
        super(SaliconNet, self).__init__()

        # backbone = self.__vgg16_backbone()
        backbone = self.__resnet34_backbone()
        self.coarse_backbone = backbone
        self.fine_backbone = copy.deepcopy(backbone)
        self.conv1x1 = torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0)

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

        return x


        