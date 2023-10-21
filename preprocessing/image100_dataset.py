class ImageNet100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, folders, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder in folders:
            folder_path = os.path.join(root_path, folder)
            for class_dir in os.listdir(folder_path):
                class_path = os.path.join(folder_path, class_dir)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_folders = [f'train.X{i}' for i in range(1, 5)]
val_folder = ['val.X']

train_dataset = ImageNet100Dataset(root_path, train_folders, transform=transform)
val_dataset = ImageNet100Dataset(root_path, val_folder, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
