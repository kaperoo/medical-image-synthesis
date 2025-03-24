import os
import argparse
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Tuple

sys.path.append(sys.argv[0])

from vae import Autoencoder, Encoder, Decoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train an autoencoder model")
    parser.add_argument("--data_dir", type=str, default="../../../data/augmented_data", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./autocheckpoints", help="Directory to save model")
    parser.add_argument("--img_size", type=Tuple, default=(128,352), help="Image size")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--z_channels", type=int, default=4, help="Number of channels in embedding space")
    parser.add_argument("--emb_channels", type=int, default=4, help="Number of dimensions in quantized embedding space")
    return parser.parse_args()

def reconstruction_loss(prediction, target):
    # L1 loss
    return F.l1_loss(prediction, target)

def kl_divergence_loss(mean, log_var):
    # KL divergence loss
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=[1, 2, 3]).mean()

def visualize_reconstruction(model, data_loader, device, save_path):
    """Visualize a batch of reconstructions"""
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        images = batch[0].to(device)
        
        # Get reconstructed images
        distribution = model.encode(images)
        z = distribution.sample()
        reconstructed = model.decode(z)
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # Original images
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            axes[0, i].imshow(img.clip(0, 1))
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")
            
            # Reconstructed images
            rec_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
            rec_img = (rec_img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            axes[1, i].imshow(rec_img.clip(0, 1))
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and transform dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((img_size[0], img_size[1])),
        transforms.Resize((args.img_size[0], args.img_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # You'll need to adjust this based on your specific dataset
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model components
    encoder = Encoder(
        channels=128,
        channel_multipliers=[1, 2, 4],
        n_resnet_blocks=2,
        in_channels=args.in_channels,
        z_channels=args.z_channels
    )
    
    decoder = Decoder(
        channels=128,
        channel_multipliers=[1, 2, 4],
        n_resnet_blocks=2,
        out_channels=args.in_channels,
        z_channels=args.z_channels
    )
    
    # Create autoencoder
    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        emb_channels=args.emb_channels,
        z_channels=args.z_channels
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_rec_loss = 0
        total_kl_loss = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            images = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Encode images to get distribution
            distribution = model.module.encode(images)
            
            # Sample from the distribution
            z = distribution.sample()
            
            # Decode the sampled embedding
            reconstructed = model.decode(z)
            
            # Calculate losses
            rec_loss = reconstruction_loss(reconstructed, images)
            kl_loss = kl_divergence_loss(distribution.mean, distribution.log_var)
            
            # Total loss with weighting
            kl_weight = 1e-6  # KL weight can be adjusted
            loss = rec_loss + kl_weight * kl_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "Rec Loss": rec_loss.item(),
                "KL Loss": kl_loss.item()
            })
        
        # Log average losses for the epoch
        avg_rec_loss = total_rec_loss / len(data_loader)
        avg_kl_loss = total_kl_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Rec Loss: {avg_rec_loss:.6f}, Avg KL Loss: {avg_kl_loss:.6f}")
        
        # Visualize reconstructions every 5 epochs
        if (epoch + 1) % 5 == 0:
            vis_path = os.path.join(args.save_dir, f"reconstruction_epoch_{epoch+1}.png")
            visualize_reconstruction(model, data_loader, device, vis_path)
        
        # # Save model checkpoint
        # if (epoch + 1) % 10 == 0:
        #     checkpoint_path = os.path.join(args.save_dir, f"autoencoder_epoch_{epoch+1}.pt")
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'rec_loss': avg_rec_loss,
        #         'kl_loss': avg_kl_loss
        #     }, checkpoint_path)
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, "autoencoder.pth"))

if __name__ == "__main__":
    args = parse_args()
    train(args)