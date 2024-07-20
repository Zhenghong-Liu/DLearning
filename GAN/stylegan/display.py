import torch
from config import device, z_dimensions, mapping_network, generator
from loops import generate_noise

from torchvision.utils import make_grid, save_image
from pathlib import Path

load_models_directory = Path("./models/stylegan/128")
target_epoch = 47
number_images = 32
resolution = 128
alpha = 1 #不知道这里的alpha有没有什么讲究，我想改但是参考代码写的是1

mapping_network.load_state_dict(
    torch.load(
        Path.joinpath(load_models_directory, f"mapping_network_{target_epoch}.pth"),
    )
)

generator.load_state_dict(
    torch.load(
        Path.joinpath(load_models_directory, f"generator_{target_epoch}.pth"),
    )
)

sample_noise = generate_noise(number_images, z_dimensions, device)
sample_w = mapping_network(sample_noise)
sample_image = generator(sample_w, resolution, alpha)
sample_image = (sample_image + 1) / 2 #因为生成器最后Tanh激活函数的原因，所以要将图像的像素值从[-1, 1]转换到[0, 1]

grid = make_grid(sample_image, nrow=8, normalize=True)

save_image(grid, f"sample_images_{target_epoch}epoch.png")

