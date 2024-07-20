import torch
from Generator import MappingNetwork, Generator
from Discriminator import Discriminator

from pathlib import Path

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

z_dimensions = 512
w_dimensions = 512


batches_per_epoch = 800 #每个epoch有800个batch
epochs_per_double = 48 #每个分辨率的训练周期
current_epoch = 0
current_doubles = 0
maximum_doubles = 3
alpha_recovery_epochs = epochs_per_double // 2 #alpha恢复的周期

load_models_from_epoch = None
load_models_directory = Path("./models/stylegan/xxx")
save_models_base_directory = Path("./models/stylegan")


mapping_network = MappingNetwork(z_dimensions, w_dimensions).to(device)
generator = Generator(w_dimensions, image_resolution=16 * 2**maximum_doubles, start_channels=512).to(device)
discriminator = Discriminator(image_resolution=16 * 2**maximum_doubles, max_channels=512).to(device)