import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import Places365
from tqdm import tqdm

from colorizers import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your data transformations
transform = Compose([
    Resize((256, 256)),  # Adjust the size as needed
    ToTensor(),
    # Add more transforms if necessary
])

# Assuming you have a DataLoader and dataset for your training and validation data
train_dataset = Places365(root='/home/legomi/Medve/colorization/imgs/places365', split='train-standard', transform=transform)
val_dataset = Places365(root='/home/legomi/Medve/colorization/imgs/places365', split='val', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoader for training and validation
batch_size = 64  # Set your desired batch size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instantiate your model and move it to GPU
model = ECCVGenerator().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer (modify this based on your preferences)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
num_epochs = 500

# TensorBoard setup
writer = SummaryWriter()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        # Move inputs and targets to GPU
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

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
torch.save(model.state_dict(), 'your_model.pth')

# Close the TensorBoard writer
writer.close()
