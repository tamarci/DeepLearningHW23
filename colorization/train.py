from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale, ToPILImage, Lambda
from torchvision import transforms
from tqdm import tqdm
import PIL

import colorspacious as cs

from colorizers import *

conf = {
    "resolution": 16,   #Image resolution, def 256
    "fraction": 0.2,    #Fraction of the dataset
    "num_workers": 12,  #For the dataloader, default 12 for 3050
    "batch_size": 1024,
    "num_epochs": 64
}
conf=OmegaConf.create(conf)

# Check if GPU is available
device_type = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device_type)
print("Using device: "+(device_type))

# Define your data transformations
transform = Compose([
    Resize((conf.resolution, conf.resolution)),  # Adjust the size as needed
    ToTensor(),
    #Lambda(rgb_to_l),
])

# Load the full dataset
full_dataset = MyPlaces365(root='./imgs/places365', split='train-standard', transform=transform)

# Calculate the number of samples to keep (e.g., 10%)
num_samples = int(len(full_dataset) * conf.fraction)

# Create a subset of the dataset
subset_dataset, _ = random_split(full_dataset, [num_samples, len(full_dataset) - num_samples])

# Split the dataset into training and validation sets
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

# Instantiate your model and move it to GPU
model = ECCVGenerator().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer (modify this based on your preferences)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard setup
writer = SummaryWriter()

# Training loop
for epoch in range(conf.num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{conf.num_epochs}')):
        # Move inputs and targets to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Ensure the data types are compatible
        outputs = outputs.float()
        targets = targets.float()

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f'Training Loss: {average_loss}')

    # Logging to TensorBoard
    writer.add_scalar('Training Loss', average_loss, epoch)

    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_targets in val_dataloader:
            # Move val_inputs and val_targets to GPU
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()

        average_val_loss = val_loss / len(val_dataloader)
        print(f'Validation Loss: {average_val_loss}')

        # Logging to TensorBoard
        writer.add_scalar('Validation Loss', average_val_loss, epoch)

# Save the trained model if needed
torch.save(model.state_dict(), 'colorization_model.pth')

# Close the TensorBoard writer
writer.close()
