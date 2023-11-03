import torch
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create the data directory if it does not exist
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Define dataset size and dimensions
num_samples = 100  # Number of samples in each set (train and validation)
patch_size = (64, 64, 28)  # Dimensions of the MRI patch
num_classes = 2  # For binary classification

# Generate synthetic MRI data
train_data = np.random.rand(num_samples, *patch_size).astype(np.float32)
val_data = np.random.rand(num_samples, *patch_size).astype(np.float32)

# Generate synthetic labels
train_labels = np.random.randint(num_classes, size=num_samples)
val_labels = np.random.randint(num_classes, size=num_samples)

# Convert to PyTorch tensors
train_data_tensor = torch.tensor(train_data).unsqueeze(1)  # Add channel dimension
train_labels_tensor = torch.tensor(train_labels).long()
val_data_tensor = torch.tensor(val_data).unsqueeze(1)  # Add channel dimension
val_labels_tensor = torch.tensor(val_labels).long()

# You can now use these tensors directly in your data loaders
torch.save(train_data_tensor, os.path.join(data_dir, 'train_data_tensor.pt'))
torch.save(train_labels_tensor, os.path.join(data_dir, 'train_labels_tensor.pt'))
torch.save(val_data_tensor, os.path.join(data_dir, 'val_data_tensor.pt'))
torch.save(val_labels_tensor, os.path.join(data_dir, 'val_labels_tensor.pt'))