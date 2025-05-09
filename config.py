import torchvision.transforms.v2 as T

transforms = {
    "train": T.Compose(
        [
            T.Resize((1024,1024)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ]
    ),
}