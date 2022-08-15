
import torch

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1=torch.nn.Conv2d(in_channels, out_channels, kernel_size =3, stride=stride, padding=1)
        self.bn1=torch.nn.BatchNorm2d(out_channels)
        self.relu=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2=torch.nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                torch.nn.BatchNorm2d(out_channels)                
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        x = x + identity
        output = self.relu(x)

        return output

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

def make_layer(in_channels, out_channels, stride=1):
    layers = []

    block = BasicBlock
    num_layer=2
    layers.append(
        block(in_channels, out_channels, stride=stride)
    )
    in_channels = out_channels * block.expansion
    for _ in range(1, num_layer):
        layer = block(in_channels, out_channels, stride=1)
        layers.append(layer)
        in_channels = block.expansion * out_channels
    return torch.nn.Sequential(*layers)

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.init = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.down1 = make_layer(64, 64)
        self.down2 = make_layer(64, 128, 2)
        self.down3 = make_layer(128, 256, 2)
        self.down4 = make_layer(256, 512, 2)
        self.down5 = make_layer(512, 1024, 2)

    def forward(self, x):
        x0 = self.init(x)

        x1 = self.down1(x0) 
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)    

        return x1, x2, x3, x4, x5

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = DeConv(1024, 512)
        self.conv1 = make_layer(1024, 512)
        self.up2 = DeConv(512, 256)
        self.conv2 = make_layer(512, 256)
        self.up3 = DeConv(256, 128)
        self.conv3 = make_layer(256, 128)
        self.up4 = DeConv(128, 64)
        self.conv4 = make_layer(128, 64)

        self.conv1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x1, x2, x3, x4, x5):
        
        x6 = self.up1(x5)
        
        x7 = torch.cat([x4, x6], dim=1)
        x8 = self.conv1(x7)
        x9 = self.up2(x8)
        
        x10 = torch.cat([x3, x9], dim=1)
        x11 = self.conv2(x10)
        x12 = self.up3(x11)

        x13 = torch.cat([x2, x12], dim=1)
        x14 = self.conv3(x13)
        x15 = self.up4(x14)

        x16 = torch.cat([x1, x15], dim=1)
        x17 = self.conv4(x16)
        x18 = self.conv1x1(x17)
    
        return x18


class ResUNet(torch.nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self, x):

        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        
        return x
