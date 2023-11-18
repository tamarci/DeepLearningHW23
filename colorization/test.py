import torch
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the image
image_path = '/home/legomi/Medve/colorization/imgs/ILSVRC2012_val_00046524.JPEG'  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')

# Convert to LAB color space using ToPILImage and ToTensor
to_pil = ToPILImage(mode="LAB")
to_tensor = ToTensor()

# Convert PIL Image to PyTorch Tensor
image_tensor = to_tensor(image)

# Convert to LAB color space
lab_image = to_pil(image_tensor)

# Print the shape of the LAB image
print(f'Shape of LAB image: {lab_image.shape}')

# Visualize the original and LAB images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original Image
axes[0].imshow(np.asarray(image))
axes[0].axis('off')
axes[0].set_title('Original Image')

# LAB Image
axes[1].imshow(lab_image.numpy())
axes[1].axis('off')
axes[1].set_title('LAB Image')

plt.show()
