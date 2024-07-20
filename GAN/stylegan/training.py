import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import os
import time
from pathlib import Path


from ffhq_dataset import FFHQ
from loops import generate_noise, train_discriminator, train_generator
from config import *


def load_epoch(mapping_network, generator, discriminator, epoch, model_directory):
    mapping_network.load_state_dict(
        torch.load(
            Path.joinpath(model_directory, f"mapping_network_{epoch}.pth"),
        )
    )

    generator.load_state_dict(
        torch.load(
            Path.joinpath(model_directory, f"generator_{epoch}.pth"),
        )
    )

    discriminator.load_state_dict(
        torch.load(
            Path.joinpath(model_directory, f"discriminator_{epoch}.pth"),
        )
    )


def save_epoch(mapping_network, generator, discriminator, epoch, model_directory):
    torch.save(
        mapping_network.state_dict(),
        Path.joinpath(model_directory, f"mapping_network_{epoch}.pth"),
    )
    torch.save(
        generator.state_dict(),
        Path.joinpath(model_directory, f"generator_{epoch}.pth"),
    )
    torch.save(
        discriminator.state_dict(),
        Path.joinpath(model_directory, f"discriminator_{epoch}.pth"),
    )

#由于generator最后激活函数是tanh的原因，需要调整一下图像的像素值
def adjust_images(images):
    return (images + 1) / 2


resize128 = transforms.Resize((128, 128), transforms.InterpolationMode.NEAREST_EXACT, antialias=True) #NEAREST_EXACT是最近邻插值，antialias=True是抗锯齿
def generate_adjusted_sample_images_128(mapping_network, generator, sample_noise, resolution, alpha):
    sample_w = mapping_network(sample_noise)
    sample_images = generator(sample_w, resolution, alpha)
    adjust_sample_images = adjust_images(sample_images)
    return resize128(adjust_sample_images)


transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(0.5, 0.5)
])

ffhq_dataset = FFHQ(transform=transformations)
ffhq_dataloader = DataLoader(
    ffhq_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

if load_models_from_epoch != None:
    load_epoch(mapping_network, generator, discriminator, load_models_from_epoch, load_models_directory)

generator_optimizer = torch.optim.RMSprop([
    {"params": mapping_network.parameters(), "lr": 0.00001},
    {"params": generator.parameters()}
], lr=0.001)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.001)

penalty_factor = 10

if current_doubles == 0:
    alpha = 1
else:
    alpha = current_epoch / alpha_recovery_epochs
    if (alpha > 0.99999): alpha = 1
alpha_difference = 1 / (alpha_recovery_epochs * batches_per_epoch)

sample_noise = generate_noise(batch_size, z_dimensions, device)

while current_doubles <= maximum_doubles:

    current_resolution = 16 * 2 ** current_doubles #最小从16x16开始训练
    resizer = transforms.Resize((current_resolution, current_resolution), antialias=True)
    save_model_directory = Path.joinpath(save_models_base_directory, str(current_resolution))
    os.makedirs(save_model_directory, exist_ok=True)

    wirter = SummaryWriter("./run/STYLEGAN/STYLEGAN_" + current_resolution + "_" + str(int(time.time())))

    while current_epoch < epochs_per_double or current_doubles == maximum_doubles: # 对于最后一个分辨率，就是死循环了，会一直训练下去。
        print(f"Epoch {current_epoch}, Resolution {current_resolution}, Alpha {alpha}")

        track_generator_losses = []
        track_real_batch_scores = []
        track_fake_batch_scores = []
        track_discriminator_losses = []

        mapping_network.train()
        generator.train()
        discriminator.train()
        for i, real_batch in enumerate(ffhq_dataloader):
            if (i == batches_per_epoch) : break #每个epoch只训练batch_per_epoch个batch

            real_batch = real_batch.to(device)
            real_batch = resizer(real_batch) #对于不同的分辨率，需要resize一下

            real_batch_scores, fake_batch_scores, discriminator_loss = train_discriminator(
                mapping_network,
                generator,
                discriminator,
                real_batch,
                discriminator_optimizer,
                penalty_factor,
                batch_size,
                current_resolution,
                alpha
            )

            generator_loss = train_generator(
                mapping_network,
                generator,
                discriminator,
                generator_optimizer,
                batch_size,
                current_resolution,
                alpha
            )

            track_generator_losses.append(generator_loss)
            track_real_batch_scores.append(real_batch_scores)
            track_fake_batch_scores.append(fake_batch_scores)
            track_discriminator_losses.append(discriminator_loss)

            del discriminator_loss
            del generator_loss

            if alpha >= 0.99999:
                alpha = 1
            else:
                alpha += alpha_difference

        mapping_network.eval()
        generator.eval()
        discriminator.eval()

        mean_real_batch_scores = torch.tensor(track_real_batch_scores).mean().item()
        mean_fake_batch_scores = torch.tensor(track_fake_batch_scores).mean().item()
        mean_generator_losses = torch.tensor(track_generator_losses).mean().item()
        mean_discriminator_losses = torch.tensor(track_discriminator_losses).mean().item()


        print(f"Mean Real Batch Scores: {mean_real_batch_scores}")
        print(f"Mean Fake Batch Scores: {mean_fake_batch_scores}")
        print(f"Mean Generator Losses: {mean_generator_losses}")
        print(f"Mean Discriminator Losses: {mean_discriminator_losses}")

        wirter.add_scalar("Loss/discriminator", mean_discriminator_losses, current_epoch) #add_scalar方法用于记录标量值，这些标量值可以在tensorboard中以图表的形式展示
        wirter.add_scalar("Loss/generator", mean_generator_losses, current_epoch)

        sample_images = generate_adjusted_sample_images_128(mapping_network, generator, sample_noise, current_resolution, alpha)
        grid = torchvision.utils.make_grid(sample_images, nrow=8, normalize=True)
        wirter.add_image("sample_images", grid, current_epoch)


























