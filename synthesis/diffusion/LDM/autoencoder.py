import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

PATH_TO_CHECKPOINT = "./checkpoints"
PATH_TO_DATA = "../../../data/augmented_data"
# IMG_SIZE = (64, 144)
IMG_SIZE = (128, 288)
BATCH_SIZE = 16

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (250, 100)
            # nn.ReLU(),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (125, 50)
            # nn.ReLU(),
            nn.SiLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),  # (125, 50)
        )

    def forward(self, x):
        return self.encoder(x)

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),  # (125, 50)
            # nn.ReLU(),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (250, 100)
            # nn.ReLU(),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (500, 200)
            # nn.Sigmoid()  # Normalize output to (0,1)
            nn.Tanh()  # Normalize output to (-1,1)
        )

    def forward(self, z):
        return self.decoder(z)

# Combine Encoder and Decoder into Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x


def adjust_image_size(img_size):
    num_downsample = 4
    min_size = 2**num_downsample
    new_height = ((img_size[0] + min_size - 1) // min_size) * min_size
    new_width = ((img_size[1] + min_size - 1) // min_size) * min_size
    return (new_height, new_width)

def load_transformed_dataset(img_size=IMG_SIZE):
    data_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size[0], img_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = torchvision.datasets.ImageFolder(PATH_TO_DATA, transform=data_transform)
    return dataset

if __name__ == "__main__":
    data = load_transformed_dataset(adjust_image_size(IMG_SIZE))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    # Training Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 4
    epochs = 50
    lr = 0.001

    # Initialize model, loss, optimizer
    autoencoder = Autoencoder(latent_dim).to(device)
    # criterion = nn.MSELoss()  # Reconstruction loss
    criterion = nn.L1Loss()  # Reconstruction loss
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)

            optimizer.zero_grad()
            recon_images = autoencoder(images)
            loss = criterion(recon_images, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save model
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    print("Training complete. Model saved.")
