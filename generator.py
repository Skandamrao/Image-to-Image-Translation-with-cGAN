import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        self.down1 = self.contract_block(input_channels, features, 4, 2, 1)
        self.down2 = self.contract_block(features, features*2, 4, 2, 1)
        self.down3 = self.contract_block(features*2, features*4, 4, 2, 1)
        self.down4 = self.contract_block(features*4, features*8, 4, 2, 1)
        self.up1 = self.expand_block(features*8, features*4, 4, 2, 1)
        self.up2 = self.expand_block(features*8, features*2, 4, 2, 1)
        self.up3 = self.expand_block(features*4, features, 4, 2, 1)
        self.final = nn.ConvTranspose2d(features*2, output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def contract_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        out = self.final(torch.cat([u3, d1], 1))
        return self.tanh(out)
