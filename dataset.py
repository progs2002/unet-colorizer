import os 

from torchvision.datasets import ImageNet
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import transforms

#bash: $kaggle datasets download arnaud58/landscape-pictures
#bash: $kaggle datasets download ambityga/imagenet100

class PairedDataset(Dataset):
    def __init__(self, root_dir, extensions=(".jpg", ".jpeg", ".png")):
        self.root_dir = root_dir
        self.transform = transforms["train"]
        self.extensions = extensions
        self.image_paths = self._load_images()

    
    def _rgb_to_grayscale(self, tensor_img):
        # Assumes input is [3, H, W] in RGB order
        r, g, b = tensor_img[0], tensor_img[1], tensor_img[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(0)

    def _load_images(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(self.extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gray = self._rgb_to_grayscale(image)
        return gray, image

if __name__ == "__main__":
    ds = PairedDataset(root_dir="/Users/progs/Pictures")
    loader = DataLoader(ds, batch_size=1)

    for gray_img, color_img in loader:
        print(color_img.shape, end=", ")
        print(gray_img.shape)