import torchvision.transforms.v2 as T
from torch import nn
import torch.nn.functional as F
import random

class MyCustomTransform(nn.Module):
    def forward(self, t): 
        random_size = random.randint(128, 512)
        t = F.interpolate(t, (random_size,random_size), mode='bilinear')
        t = F.interpolate(t, (512,512), mode='bilinear')
        return t

transforms = {
    "train": T.Compose(
        [
            T.Resize((512,512)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            MyCustomTransform()
        ]
    ),
}