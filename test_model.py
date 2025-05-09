import torch
from models.unet import PBPUNet

if __name__ == "__main__":
    model = PBPUNet(1,3)
    x = torch.randn(1,1,512,512)
    with torch.no_grad():
        out = model(x)
    print(out.shape)