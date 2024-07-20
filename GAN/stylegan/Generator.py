import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from itertools import pairwise


#==============================================================================
# 自定义线形层
# 这是为了添加一个scale，从而标准化输入的方差。
# 为了把bias也进行标准化，因此需要把bias单独拿出来，把本来的linear.bias设为None
# 这个类中，self.scale是常数，不能被训练优化。（这个也不需要优化）
# self.bias来自于self.linear.bias，仍然属于nn.Parameter，因此可以被训练优化。
#==============================================================================
class CLinear(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.linear = nn.Linear(input, output)
        self.scale = (2 / input) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias * self.scale


#==============================================================================
# 自定义卷积层, 与CLinear类似
#==============================================================================
class CConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.scale = (2 / (input_channel * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(-1, 1, 1) * self.scale #bias需要和x的维度一致

class CConvTranspose2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding)
        self.scale = (2 / (input_channel * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x_and_w):
        x, w= x_and_w
        return self.conv(x * self.scale) + self.bias.view(-1, 1, 1) * self.scale, w

class InterpolateWrapper(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x_and_w):
        x, w = x_and_w
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode), w


class MappingNetwork(nn.Module):
    def __init__(self, z_dimensions, w_dimensions, rms_norm_epsilon = 1e-6):
        super().__init__()
        self.rms_norm_epsilon = rms_norm_epsilon
        blocks = []
        for i in range(7):
            blocks += [
                CLinear(z_dimensions, z_dimensions),
                nn.LeakyReLU(0.2)
            ]
        blocks += [CLinear(z_dimensions, w_dimensions)]
        self.mapping = nn.Sequential(*blocks)

    def forward(self, z):
        # RMS Norm
        # RMS Norm归一化方法是：z' = z / sqrt(mean(z^2) + epsilon)
        z = z * (torch.mean(z**2, dim=1, keepdim=True) + self.rms_norm_epsilon).rsqrt() #rsqrt()是开根号的倒数
        return self.mapping(z)
    
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, w_dimensions, channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_network = CLinear(w_dimensions, channels)
        self.style_shift_network = CLinear(w_dimensions, channels)

    def forward(self, x_and_w): #使用设计模式：流水线模式，要使得每个Block接收到的输入和输出都是一样的，这里输入输出都是一个x_and_w元组
        x, w = x_and_w
        x = self.instance_norm(x)
        # style_scale = self.style_scale_network(w)[:, :, None, None] #这行的效果和下面的一样
        style_scale = self.style_scale_network(w).unsqueeze(2).unsqueeze(3)
        style_shift = self.style_shift_network(w).unsqueeze(2).unsqueeze(3)

        return x * style_scale + style_shift, w

class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_parameter("scale", nn.Parameter(torch.zeros([channels, 1, 1]))) #后面相乘的时候，会利用广播机制

    def forward(self, x_and_w):
        x, w = x_and_w
        noise = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
        return x + noise * self.scale, w

class LeakyReluWrapper(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x_and_w):
        x, w = x_and_w
        return self.leaky_relu(x), w


class ConvNoiseNorm(nn.Module):
    def __init__(self, in_channels, out_channels, w_dimensions):
        super().__init__()
        self.conv = CConv2d(in_channels, out_channels, 3, 1, 1)
        self.noise_relu_norm = nn.Sequential(
            InjectNoise(out_channels),
            LeakyReluWrapper(),
            AdaptiveInstanceNorm(w_dimensions, out_channels)
        ) #这个pipeline最后的输出还是x_and_w元组

    def forward(self, x_and_w):
        x, w = x_and_w
        x = self.conv(x)
        return self.noise_relu_norm((x, w))


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dimensions, first=False, isInterpolate=True):
        super().__init__()
        if first:
            self.model = nn.Sequential(
                InjectNoise(in_channels),
                LeakyReluWrapper(0.2),
                AdaptiveInstanceNorm(w_dimensions, in_channels),
                ConvNoiseNorm(in_channels, out_channels, w_dimensions)
            )
        else:
            upsample = InterpolateWrapper(2, "bilinear") if isInterpolate else CConvTranspose2d(in_channels, in_channels, 4, 2, 1)
            self.model = nn.Sequential(
                upsample,
                ConvNoiseNorm(in_channels, out_channels, w_dimensions),
                ConvNoiseNorm(out_channels, out_channels, w_dimensions)
            )

    def forward(self, x_and_w):
        x, w = x_and_w
        return self.model((x, w))


class Generator(nn.Module):
    def __init__(self, w_dimensions, image_resolution, image_channels=3, start_resolution=4, start_channels=512):
        super().__init__()
        self.start_resolution = start_resolution
        self.double_required = int(log2(image_resolution)) - int(log2(start_resolution))

        #生成器部分，最初输入是常数，但是这个常数应该是可以学习的
        self.register_parameter("start_constant", nn.Parameter(torch.ones([start_channels, start_resolution, start_resolution])))

        self.initial = SynthesisBlock(start_channels, start_channels, w_dimensions, first=True)
        channels = [min(start_channels, 64*2**i) for i in range(self.double_required, -1, -1)] #这里的channels是一个list，从大到小，一直降到64
        channels = [start_channels] + channels #补全第一个start_channels

        self.post_upsample_list = nn.ModuleList()
        self.rgb_converter_list = nn.ModuleList()
        for i, (input_channel, output_channel) in enumerate(pairwise(channels)):
            self.post_upsample_list.append(
                SynthesisBlock(input_channel, output_channel, w_dimensions, isInterpolate=i < 4)
            )
            self.rgb_converter_list.append(
                nn.Sequential(
                    nn.Conv2d(output_channel, image_channels, 1),
                    nn.Tanh()
                )
            )

    def forward(self, w, resolution, alpha):
        current_doubles_required = int(log2(resolution)) - int(log2(self.start_resolution))
        assert 0 <= current_doubles_required <= self.double_required , "Resolution not supported by the generator"

        x = self.start_constant.repeat([w.shape[0], 1, 1, 1])
        x, w = self.initial((x, w))

        for post_upsamle in self.post_upsample_list[:current_doubles_required - 1]:
            x, w = post_upsamle((x, w)) #流水线模式，各个模块传输数据

        #===============================================================
        # 最后一个block的输出，是特别的，需要单独处理
        # 最后一个block需要输出两个相同分辨率的图像
        # 这两个图像中，一个进行了最后一次的style和noise的注入，另一个没有
        #===============================================================

        upscaled_x = F.interpolate(x, scale_factor=2, mode="bilinear")
        upscaled_x_rgb = self.rgb_converter_list[current_doubles_required - 2](upscaled_x) #由于双线性差值，所以channel是上一个block的channel

        x, w = self.post_upsample_list[current_doubles_required - 1]((x, w))
        x_rgb = self.rgb_converter_list[current_doubles_required - 1](x)

        #torch.lerp(star, end, weight) : 返回结果是out= star t+ (end-start) * weight
        #这里的alpha是一个权重，用于控制两个图像的混合
        #low alpha = new image less weight, high alpha = new image more weight
        return torch.lerp(upscaled_x_rgb, x_rgb, alpha) #相当于最后一层是一个为了计算一个残差，并给了一个值为alpha的权重


if __name__ == '__main__':
    w_dimensions = 512
    z_dimensions = 512
    image_resolution = 1024
    image_channels = 3
    start_resolution = 4
    start_channels = 512


    generator = Generator(w_dimensions, image_resolution, image_channels, start_resolution, start_channels)
    mapper = MappingNetwork(z_dimensions, w_dimensions)


    z = torch.randn([1, z_dimensions])
    w = mapper(z)
    resolution = 1024
    alpha = 0.5
    image = generator(w, resolution, alpha)
    print(image.shape)


























