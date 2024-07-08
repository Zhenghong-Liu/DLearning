
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalization:
                block.append(nn.InstanceNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), # ZeroPad2d((left, right, top, bottom))
            nn.Conv2d(512, 1, 4, padding=1, bias=False), #判别器最后一层不用bias，可以和bn层配合使用，保持数值稳定和简化实现
        )

    def forward(self, img):
        return self.model(img)
