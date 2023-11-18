from torch.utils.data import Dataset
from torchvision.datasets import Places365

class MyPlaces365(Dataset):
    def __init__(self, root, split='train-standard', transform=None,target_transform=None,download=False):
        self.dataset = Places365(root=root, split=split, transform=transform, download=download)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Return the image as the target (instead of label)
        return img, img
