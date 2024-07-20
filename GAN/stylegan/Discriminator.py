import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from itertools import pairwise

from Generator import CConv2d, CLinear

class Discriminator(nn.Module):
    def __init__(self, image_resolution, image_channels=3, max_channels=512):
        super().__init__()
        self.image_resolution_log2 = int(log2(image_resolution))
        channels = [min(64 * 2 **i, max_channels) for i in range(self.image_resolution_log2)] #判别起是downsample，channels是依次增加的
        channels = [image_channels] + channels

        self.rgb_converter_list = nn.ModuleList()
        self.pre_downsample_list = nn.ModuleList()

        for i, (in_channels, out_channels) in enumerate(pairwise(channels[:-2])):  #这里的-2是因为最后的图像分辨率要是4x4，所以不需要再downsample
            self.rgb_converter_list.append(
                nn.Sequential(
                    CConv2d(image_channels, in_channels, 3,1,1),
                    nn.LeakyReLU(0.2)
                )
            )

            self.pre_downsample_list.append( #和参考代码不同，我对模块做了一些调整
                nn.Sequential(
                    CConv2d(in_channels, out_channels, 3, 2, 1),
                    nn.LeakyReLU(0.2),
                    CConv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2)
                )
            )

            self.final = nn.Sequential(
                CConv2d(channels[-1], channels[-1], 3, 1, 1),
                nn.Flatten(),
                CLinear(channels[-1] * 4 * 4, channels[-1]),
                nn.LeakyReLU(0.2),
                CLinear(channels[-1], 1),
            )

    def forward(self, x, resolution, alpha):
        resolution_log2 = int(log2(resolution))
        assert resolution_log2 <= self.image_resolution_log2, "resolution should be smaller than image_resolution"
        start_index = self.image_resolution_log2 - resolution_log2

        half_x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        half_x = self.rgb_converter_list[start_index + 1](half_x) #这里加一是为了让通过线性差值的channles和通过卷积的下采样的channels一致

        x = self.rgb_converter_list[start_index](x)
        x = self.pre_downsample_list[start_index](x)

        x = torch.lerp(half_x, x, alpha) #torch.lerp(star, end, weight) : 返回结果是out= start+ (end-start) * weight

        for pre_downsample in self.pre_downsample_list[start_index + 1:]:
            x = pre_downsample(x)
        return self.final(x)





if __name__ == '__main__':

    test_images = [torch.randn([1, 3, 2 ** i, 2 ** i]) for i in range(4, 11)]
    discriminator = Discriminator(1024)
    for i, image in enumerate(test_images):
        resolution = 2 ** (i + 4)
        result = discriminator(image, resolution, 0.5)
        print(f"Input shape: {image.shape}, output shape: {result.shape}, output value: {result.item()}")