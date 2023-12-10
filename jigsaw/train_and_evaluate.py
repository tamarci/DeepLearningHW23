import os
from zipfile import ZipFile
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import Places365, ImageFolder
from torchvision.datasets.utils import download_url
from torchvision import transforms
from lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm

seed_everything(42)
np.random.seed(42)

CONV1_C = 96
CONV2_C = 256
CONV3_C = 384
FC1 = 1024 
FC2 = 256
N_PERMUTATIONS = 24
N_TILES = 2
IMG_SIZE = N_TILES * 64

# utility classes
class TinyImageNetDataModule(LightningDataModule):
    def __init__(
        self, downloaded_zip="tiny-imagenet-200.zip", data_folder="data", batch_size=32, num_workers=8
    ):
        super().__init__()
        self.downloaded_zip = downloaded_zip
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.isfile(self.downloaded_zip):
            # download if needed
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            os.makedirs(self.data_folder, exist_ok=True)
            download_url(url, self.data_folder)

        if not os.path.isdir(os.path.join(self.data_folder, "tiny-imagenet-200")):
            # extract if not yet extracted
            with ZipFile(os.path.join(self.data_folder, "tiny-imagenet-200.zip"), "r") as zip_ref:
                zip_ref.extractall(self.data_folder)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imageNet standard
        ])

        self.train = ImageFolder(os.path.join(self.data_folder, "tiny-imagenet-200/train"), transform=transform)
        self.val = ImageFolder(os.path.join(self.data_folder, "tiny-imagenet-200/val"), transform=transform)
        self.test = ImageFolder(os.path.join(self.data_folder, "tiny-imagenet-200/test"), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

class Places365TileDataModule(LightningDataModule):
    def __init__(
        self,
        places_folder="data/places365",
        permutations_file="data/naroozi_perms_100_patches_9_max.npy",
        n_permutations=N_PERMUTATIONS,
        batch_size=2,
        num_workers=8
    ):
        super().__init__()
        self.places_folder = places_folder
        self.permutations = np.load(permutations_file)
        self.permutations -= 1 # 0 based indexing
        self.n_permutations = n_permutations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imageNet standard
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        download = not os.path.isdir(self.places_folder)
        print(f"{download=}")
        
        # train is too big
        base_dataset = Places365(self.places_folder, download=download, split="val")
        # create tiles and split
        tiles_dataset = self.__get_tiles_dataset(base_dataset)
        self.train, self.val, self.test = random_split(tiles_dataset, [0.7, 0.15, 0.15])

    def __get_tiles_dataset(self, base_dataset):
        # create labels (indexes of the intended permutations)
        permutation_idxs = np.random.choice(self.n_permutations, len(base_dataset), replace=True)
        tile_label_pairs = []

        for (img, _), permutation_idx in tqdm(zip(base_dataset, permutation_idxs), desc="Loading pretrain data", total=len(base_dataset)):
            # cut tiles, create tensors and apply transform
            img = img.resize((IMG_SIZE, IMG_SIZE))
            tiles = [
                self.transform(img.crop([
                    (n // N_TILES) * IMG_SIZE // N_TILES, # xmin
                    (n % N_TILES) * IMG_SIZE // N_TILES, # ymin
                    (n // N_TILES + 1) * IMG_SIZE // N_TILES, # xmax
                    (n % N_TILES + 1) * IMG_SIZE // N_TILES, # ymax
                ])).unsqueeze(0) # add batch dimension
                for n in range(N_TILES*N_TILES)
            ]
            # permute
            permutation = self.permutations[permutation_idx]
            permuted_tiles = torch.cat(tiles, dim=0)[permutation,...]
            # append
            tile_label_pairs.append((permuted_tiles, permutation_idx))

        class TileDataset(Dataset):
            def __init__(self, tile_label_pairs):
                super().__init__()
                self.tiles = [x[0] for x in tile_label_pairs]
                self.labels = [x[1] for x in tile_label_pairs]

            def __getitem__(self, index):
                return self.tiles[index], self.labels[index]
            
            def __len__(self):
                return len(self.tiles)

        return TileDataset(tile_label_pairs)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class JigsawModel(LightningModule):
    def __init__(self, lr=1e-2):
        super().__init__()
        # model building based on global parameters
        backbone = nn.Sequential(
            # conv1
            nn.Conv2d(3, CONV1_C, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(CONV1_C),
            # pooling 1
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.LocalResponseNorm(5),
            # conv2
            nn.Conv2d(CONV1_C, CONV2_C, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(CONV2_C),
            # pooling 2
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5),
            # conv3
            nn.Conv2d(CONV2_C, CONV3_C, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
        )
        fc_tile = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV3_C*10*10, FC1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.feature_extractor = nn.Sequential(backbone, fc_tile)
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            # fc7
            nn.Linear(9*FC1, FC2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            # classifier
            nn.Linear(FC2, N_PERMUTATIONS),
        )
        
        # everything else
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # force batch size
        x = x.view(-1, N_TILES*N_TILES, 3, IMG_SIZE//N_TILES, IMG_SIZE//N_TILES)
        B = x.shape[0]
        x = x.transpose(0,1) # put tile dimension first

        # feature extractor for each tile
        x_list = []
        for i in range(N_TILES*N_TILES):
            z = self.feature_extractor(x[i])
            z = z.view(B, 1, -1) # reserve dim 1 for tile dimension - concat over this
            x_list.append(z)

        # global fc layer
        x = torch.cat(x_list, dim=1)
        x = self.fc_head(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        self.log("train loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        self.log("val loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def save_feature_extractor(self, filepath):
        torch.save(self.feature_extractor, filepath)


class LinearClassifier(LightningModule):
    def __init__(self, backbone_path, backbone_output_size=FC1, num_classes=200):
        super().__init__()
        self.feature_extractor = torch.load(backbone_path)
        self.classifier = nn.Linear(backbone_output_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        img, true_label = batch
        pred_label = self(img)
        loss = self.loss(pred_label, true_label)
        self.log("train loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, true_label = batch
        pred_label = self(img)
        loss = self.loss(pred_label, true_label)
        self.log("val loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        img, true_label = batch
        pred_label = self(img)
        loss = self.loss(pred_label, true_label)
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        # train only the linear layer
        optimizer = Adam(self.classifier.parameters(), lr=0.01)
        return optimizer


# main functions: pretrain, evaluation
def train_and_save():
    datamodule = Places365TileDataModule(batch_size=64)
    # datamodule = Places365TileDataModule(batch_size=64, permutations_file="data/perm4.npy")
    wandb_logger = WandbLogger(project="DL-HF-Jigsaw-Pretrain")
    model = JigsawModel()
    trainer = Trainer(max_epochs=5, logger=wandb_logger)
    print("Starting fit...")
    trainer.fit(model, datamodule=datamodule)
    # trainer.test(model, datamodule=datamodule) # it takes up a lot of resources, it killed my system
    model.save_feature_extractor("models/jigsaw_10_1024.pth")
    # model.save_feature_extractor("models/jigsaw_2x2_1024.pth")

# def load_feature_extractor_and_evaluate(feature_extractor_path="models/random_backbone_1024.pth"):
def load_feature_extractor_and_evaluate(feature_extractor_path="models/jigsaw_10_1024.pth"):
# def load_feature_extractor_and_evaluate(feature_extractor_path="models/jigsaw_2x2_1024.pth"):
    datamodule = TinyImageNetDataModule(batch_size=256)
    wandb_logger = WandbLogger(project="DL-HF-Linear-Benchmark")
    model = LinearClassifier(feature_extractor_path)
    trainer = Trainer(max_epochs=10, logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    train_and_save()
    load_feature_extractor_and_evaluate()