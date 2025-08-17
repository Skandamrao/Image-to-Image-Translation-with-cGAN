import torch
from torchvision.utils import save_image
from models.generator import UNetGenerator

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
    G.eval()
    sample = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        fake = G(sample)
    save_image(fake, "outputs/generated.png")

if __name__ == "__main__":
    test()
