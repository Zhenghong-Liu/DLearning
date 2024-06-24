"""
Math2Latex GPTHelpVAE.py
2024年06月15日
by Zhenghong Liu

GPT 生成的代码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(64 * 7 * 7, 128)
        self.fc3 = nn.Linear(128, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z).view(-1, 64, 7, 7)
        return self.decoder(z), mu, logvar

    def sample(self):
        z = torch.randn((1, 128), device=self.device)
        z = self.fc3(z).view(-1, 64, 7, 7)
        return self.decoder(z).detach().cpu().numpy().squeeze()

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device : {device}")

    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='../data/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')
        if (epoch + 1) % 5 == 0:
            sample_img = model.sample()
            plt.imshow(sample_img, cmap="gray")
            plt.title(f"Sample image at epoch {epoch + 1}")
            plt.show()

    print("Generating samples...")
    model.eval()
    plt.figure(figsize=(15, 15))
    with torch.no_grad():
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(model.sample(), cmap="gray")
    plt.show()

