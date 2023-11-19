import torch
from torch.utils.data import Dataset
from torchvision.datasets import Places365
from kornia.color.lab import linear_rgb_to_rgb, rgb_to_linear_rgb, rgb_to_xyz, xyz_to_rgb

#Torch does not have a RGB2LAB conversion, we used the code of kornia: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html

class MyPlaces365(Dataset):
    def __init__(self, root, split='train-standard', transform=None,download=False):
        self.dataset = Places365(root=root, split=split, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)

        # Convert from sRGB to Linear RGB
        lin_rgb = rgb_to_linear_rgb(img)

        xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

        # normalize for D65 white point
        xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
        xyz_normalized = torch.div(xyz_im, xyz_ref_white)

        threshold = 0.008856
        power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
        scale = 7.787 * xyz_normalized + 4.0 / 29.0
        xyz_int = torch.where(xyz_normalized > threshold, power, scale)

        x: torch.Tensor = xyz_int[..., 0, :, :]
        y: torch.Tensor = xyz_int[..., 1, :, :]
        z: torch.Tensor = xyz_int[..., 2, :, :]

        L: torch.Tensor = (116.0 * y) - 16.0
        a: torch.Tensor = 500.0 * (x - y)
        _b: torch.Tensor = 200.0 * (y - z)

        out: torch.Tensor = torch.stack([a, _b], dim=-3)

        # Return the L and the AB as target
        return L.unsqueeze(0), out
