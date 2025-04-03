import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from scipy.linalg import sqrtm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

REAL_PATH = "resized_data"
FAKE_PATH = "generated_data"

def load_images(dataset_path, batch_size=32, image_size=299):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)
    return dataset 

def load_images_grayscale(dataset_path, batch_size=32, image_size=299):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)
    return dataset

def get_features(dataloader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feat = model(images).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)

def compute_fid(real_features, fake_features):
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * sqrtm(sigma_real @ sigma_fake))
    return fid.real

def compute_is(fake_features, num_splits=10):
    p_yx = torch.softmax(torch.tensor(fake_features), dim=1)
    p_y = p_yx.mean(dim=0) 
    kl_divergences = (p_yx * (torch.log(p_yx + 1e-16) - torch.log(p_y + 1e-16))).sum(dim=1)

    is_scores = torch.exp(kl_divergences)
    split_size = len(is_scores) // num_splits
    split_scores = []

    for i in range(num_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < num_splits - 1 else len(is_scores)
        split_scores.append(is_scores[start_index:end_index].mean().item())

    return np.mean(split_scores), np.std(split_scores)

def compute_ssim(real_dataloader, fake_dataloader, device):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    real_images, fake_images = [], []
    with torch.no_grad():
        for (real, _), (fake, _) in zip(real_dataloader, fake_dataloader):
            real_images.append(real.to(device))
            fake_images.append(fake.to(device))
    return ssim(torch.cat(real_images), torch.cat(fake_images)).item()

def compute_psnr(real_dataloader, fake_dataloader, device):
    mse_sum, count = 0.0, 0
    with torch.no_grad():
        for (real, _), (fake, _) in zip(real_dataloader, fake_dataloader):
            real, fake = real.to(device), fake.to(device)
            mse = F.mse_loss(real, fake, reduction='sum').item()
            mse_sum += mse
            count += real.numel()

    mse_mean = mse_sum / count
    if mse_mean == 0:
        return float("inf") 
    
    psnr = 10 * torch.log10(torch.tensor(1.0, device=device) / mse_mean)
    return psnr.item()

def main(real_path, fake_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    real_dataset = load_images(real_path)
    fake_dataset = load_images(fake_path)
    
    min_length = min(len(real_dataset), len(fake_dataset))

    real_subset = Subset(real_dataset, range(min_length))
    fake_subset = Subset(fake_dataset, range(min_length))

    real_loader = DataLoader(real_subset, batch_size=128, shuffle=False)
    fake_loader = DataLoader(fake_subset, batch_size=128, shuffle=False)
    
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    
    real_features = get_features(real_loader, inception, device)
    fake_features = get_features(fake_loader, inception, device)
    
    fid = compute_fid(real_features, fake_features)
    is_score, is_std = compute_is(fake_features)
    
    real_dataset_gray = load_images_grayscale(real_path)
    fake_dataset_gray = load_images_grayscale(fake_path)

    real_subset_gray = Subset(real_dataset_gray, range(min_length))
    fake_subset_gray = Subset(fake_dataset_gray, range(min_length))

    real_loader_gray = DataLoader(real_subset_gray, batch_size=128, shuffle=False)
    fake_loader_gray = DataLoader(fake_subset_gray, batch_size=128, shuffle=False)

    ssim_score = compute_ssim(real_loader_gray, fake_loader_gray, device)
    psnr_score = compute_psnr(real_loader, fake_loader, device)

    print(f"FID: {fid:.2f}, IS: {is_score:.2f} +- {is_std:.2f}, SSIM: {ssim_score:.2f}, PSNR: {psnr_score:.2f}")

if __name__ == "__main__":
    main(REAL_PATH, FAKE_PATH)
