"""
Math2Latex UNet.py
2024年06月30日
by Zhenghong Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    "conv => BN => ReLU => conv => BN => ReLU"

    def __init__(self, input_channels, output_channels, middle_channels=None):
        super(DoubleConv, self).__init__()
        if not middle_channels: #middle_channels不给定的话就是默认和output_channels一样
            middle_channels = output_channels

        self.double_conv = nn.Sequential(
            # 卷积后面如果接norm，那么最好就把bias转为False，因为这样可以节省内存，还不影响结果。
            nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1, stride=1, bias=False), #bias默认为True
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    "使用maxpool实现下采样后再double_conv"
    def __init__(self, input_channels, output_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels, output_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    "实现上采样后再double_conv"
    def __init__(self, input_channels, output_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            #align_corners=True:每个像素的在矩阵里的下标i,j被直接视作坐标系里的一个个的坐标点(x,y)进行计算。每个像素点都是坐标上的点，而不是传统意义上的小方块。
            #如果是False的话，类似于，一个点分裂成了四个点，而True的话，是在这个点的旁边添加了3个点。
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(input_channels, output_channels, input_channels // 2) #bilinear没有改channels，可以根据调节Double_conv中的middle来改
        else:
            self.up = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        #这时候，x1和x2的长和宽，会有一点区别。x2可能会比x1大一圈
        #输入的格式是CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        #根据差距，填充padding, pad的顺序是左，右，上，下
        x1 = F.pad(x1, [diffX // 2, diffX // 2,
                        diffY // 2, diffY // 2])

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OutConv, self).__init__()
        #kernel_size=1类似于MLP，可以实现特征融合，改变通道数。
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear = False):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1 #因为在上采样的那一步，转置卷积已经降了一半的channels，但是插值的方法的channels并没有减少。
        self.down4 = Down(512, 1024//factor) #由于插值的方法没法降低通道数，所以需要在下采样的时候，就帮他降通道
        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, output_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output

    def use_checkpointing(self):
        '''
        这个技术主要用于训练时内存优化，它允许我们以计算时间为代价，减少训练深度网络时的内存占用。
        使用梯度检查点技术可以在训练大型模型时减少显存的占用，但由于在反向传播时额外的重新计算，它会增加一些计算成本
        '''
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)




if __name__ == '__main__':
    model = Unet(3, 3)
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(output.shape)

    from torchsummary import summary
    summary(model, input_size=(3, 128, 128), device='cpu')



