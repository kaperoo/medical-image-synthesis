import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys

sys.path.append(sys.argv[0])

PRINT = True
SAVE_PATH = "."
PATH_TO_CHECKPOINT = "checkpoints"
PATH_TO_DATA = "../../../data/augmented_data"
# IMG_SIZE = (64, 144)
# IMG_SIZE = (128, 288)
IMG_SIZE = (128, 352)
# IMG_SIZE = (208, 560)
BATCH_SIZE = 16

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (250, 100)
            nn.AvgPool2d(2),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (250, 100)
            nn.ReLU(),
            # nn.SiLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (125, 50)
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (125, 50)
            nn.ReLU(),
            # nn.SiLU(),
            # nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),  # (125, 50)
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),  # (125, 50)
        )

    def forward(self, x):
        return self.encoder(x)

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(latent_dim, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),  # (125, 50)
            nn.ReLU(),
            # nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (250, 100)
            nn.ReLU(),
            # nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 3, padding=1),
            # nn.BatchNorm2d(1),
            # nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (500, 200)
            nn.Tanh()  # Normalize output to (-1,1)
        )

    def forward(self, z):
        return self.decoder(z)

    # def __init__(self, latent_dim=4):
    #     super().__init__()
        
    #     self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    #     self.conv1 = nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    #     # self.activation = nn.SiLU()
    #     self.activation = nn.ReLU()
        
    # def forward(self, z):
    #     x = self.upsample(z)  # (125, 50) -> (250, 100)
    #     x = self.activation(self.conv1(x))
        
    #     x = self.upsample(x)  # (250, 100) -> (500, 200)
    #     x = self.activation(self.conv2(x))
        
    #     x = self.conv3(x)  # Final output layer
    #     x = torch.tanh(x)  # Normalize to (-1,1)
        
    #     return x

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
        # transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        transforms.Normalize([0.5], [0.5]),
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = torchvision.datasets.ImageFolder(PATH_TO_DATA, transform=data_transform)
    return dataset

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if i == 0:
                continue
            if arg == "--save":
                SAVE_PATH = sys.argv[i+1]
            elif arg == "--data":
                PATH_TO_DATA = sys.argv[i+1]
            elif arg == "-p":
                PRINT = False
    
    if PRINT:
        print(adjust_image_size(IMG_SIZE))
    data = load_transformed_dataset(adjust_image_size(IMG_SIZE))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    # Training Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 4
    epochs = 150
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
        if PRINT:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save model
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    if PRINT:
        print("Training complete. Model saved.")
