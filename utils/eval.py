import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
import numpy as np
from scipy.linalg import sqrtm
import os
from PIL import Image

REAL_PATH = 'data/'
FAKE_PATH = 'generated_data/'


def compute_inception_score(fake_images, splits=10):
    device = fake_images.device
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    transform = transforms.Resize((299, 299), antialias=True)
    fake_images_resized = transform(fake_images)

    if fake_images.shape[1] == 1:
        fake_images_resized = fake_images_resized.repeat(1, 3, 1, 1)

    with torch.no_grad():
        logits = inception(fake_images_resized).detach().cpu().numpy()
    
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()

    split_scores = []
    N = probs.shape[0]
    for k in range(splits):
        part = probs[k * (N // splits): (k + 1) * (N // splits), :]
        p_y = np.mean(part, axis=0)
        scores = [F.kl_div(torch.tensor(p_y).log(), torch.tensor(p_x), reduction="sum").item() for p_x in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def compute_fid(real_images, fake_images):
    device = real_images.device
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    transform = transforms.Resize((299, 299), antialias=True)
    
    def preprocess(imgs):
        imgs = transform(imgs)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return imgs

    real_images_resized = preprocess(real_images)
    fake_images_resized = preprocess(fake_images)

    with torch.no_grad():
        real_features = inception(real_images_resized).cpu().numpy()
        fake_features = inception(fake_images_resized).cpu().numpy()

    # Compute statistics (mean & covariance)
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Compute Frechet distance
    cov_sqrt, _ = sqrtm(sigma_real @ sigma_fake, disp=False)
    fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)

    return fid

def gaussian_kernel(window_size=11, sigma=1.5):
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss[:, None] @ gauss[None, :]
    return kernel / kernel.sum()

def gaussian_blur(img, window_size=11, sigma=1.5):
    kernel = gaussian_kernel(window_size, sigma).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(img.device, dtype=img.dtype)
    padding = window_size // 2
    return F.conv2d(img, kernel, padding=padding, groups=img.shape[1])

def gradient_magnitude(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    
    return torch.sqrt(grad_x ** 2 + grad_y ** 2)

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    mu1 = gaussian_blur(img1, window_size)
    mu2 = gaussian_blur(img2, window_size)
    
    sigma1_sq = gaussian_blur(img1 * img1, window_size) - mu1 ** 2
    sigma2_sq = gaussian_blur(img2 * img2, window_size) - mu2 ** 2
    sigma12 = gaussian_blur(img1 * img2, window_size) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def multi_scale_ssim(img1, img2, scales=4):
    """Computes MS-SSIM across multiple scales."""
    msssim = []
    for _ in range(scales):
        msssim.append(ssim(img1, img2))
        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
    
    return torch.stack(msssim).mean()

def four_scale_gradient_regularized_ssim(img1, img2, alpha=0.5):
    msssim = multi_scale_ssim(img1, img2)
    
    grad1 = gradient_magnitude(img1)
    grad2 = gradient_magnitude(img2)
    grad_sim = ssim(grad1, grad2)
    
    return alpha * msssim + (1 - alpha) * grad_sim


def load_images_from_folder(root_folder, num_samples=None):
    images, labels = [], []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    class_to_idx = {class_name: i for i, class_name in enumerate(sorted(os.listdir(root_folder)))}

    for class_name, class_id in class_to_idx.items():
        class_path = os.path.join(root_folder, class_name)
        if os.path.isdir(class_path):
            image_files = sorted(os.listdir(class_path))[:num_samples]
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert("RGB")
                images.append(transform(img))
                labels.append(class_id)

    return torch.stack(images), torch.tensor(labels)

if __name__ == "__main__":
    samples = 20

    
    print('Loading data...')
    fake_images, fake_labels = load_images_from_folder(FAKE_PATH, samples)
    real_images, real_labels = load_images_from_folder(REAL_PATH, samples)
    
    print('\nCalculating IS...')
    is_mean, is_std = compute_inception_score(fake_images)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    print('\nCalculating FID...')
    fid_score = compute_fid(real_images, fake_images)
    print(f"FID Score: {fid_score:.4f}")
    
    fake_images = transforms.Grayscale()(fake_images)
    real_images = transforms.Grayscale()(real_images)

    print('\nCalculating SSIM...')
    ssim_score = ssim(fake_images, real_images)
    print(f"SSIM Score: {ssim_score.item():.4f}")

    print('\nCalculating 4-G-R-SSIM...')
    four_gr_ssim_score = four_scale_gradient_regularized_ssim(fake_images, real_images)
    print(f"4-G-R-SSIM Score: {four_gr_ssim_score.item():.4f}")