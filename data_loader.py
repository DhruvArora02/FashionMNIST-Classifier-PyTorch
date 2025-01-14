import torch
from torchvision import datasets, transforms

def get_data_loader(training=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.FashionMNIST(
        './data',
        train=training,
        download=True,
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=training
    )
    return loader
