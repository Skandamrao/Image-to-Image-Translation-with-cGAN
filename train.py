import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator

# Dummy dataset for structure only
class DummyDataset(Dataset):
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        return torch.randn(3, 256, 256), torch.randn(3, 256, 256)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    opt_gen = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    loader = DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    for epoch in range(1):
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # Train Discriminator
            fake = G(x)
            D_real = D(x, y)
            D_fake = D(x, fake.detach())
            loss_D = (criterion_GAN(D_real, torch.ones_like(D_real)) +
                      criterion_GAN(D_fake, torch.zeros_like(D_fake))) / 2
            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()

            # Train Generator
            D_fake = D(x, fake)
            loss_G = criterion_GAN(D_fake, torch.ones_like(D_fake)) + criterion_L1(fake, y)
            opt_gen.zero_grad()
            loss_G.backward()
            opt_gen.step()

            print(f"Epoch {epoch} Batch {idx} Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

if __name__ == "__main__":
    train()
