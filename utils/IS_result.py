import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms, datasets
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

IMAGE_PATH = "utils/WGAN-GP" # Change this to your generated image folder path
# MOSTLY FROM CHATGPT!

def get_inception_score(images, batch_size=32, resize=True, splits=10):
    """
    Computes the Inception Score for a set of generated images.
    
    Args:
        images (Tensor): A tensor of shape (N, C, H, W) with pixel values in range [0, 1].
        batch_size (int): Batch size for Inception model.
        resize (bool): Whether to resize images to 299x299 for Inception model.
        splits (int): Number of splits for calculating the score.

    Returns:
        float: Inception Score mean.
        float: Inception Score standard deviation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained Inception v3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    
    # If resizing is needed for Inception
    if resize:
        up = transforms.Resize((299, 299), antialias=True)

    def preprocess(img):
        if resize:
            img = up(img)
        return img

    # Disable gradients for efficiency
    with torch.no_grad():
        preds = []
        for i in tqdm(range(0, len(images), batch_size), desc="Computing Inception Predictions"):
            batch = images[i:i + batch_size].to(device)
            batch = preprocess(batch)
            # Get softmax probabilities from Inception
            pred = F.softmax(inception(batch), dim=1).cpu().numpy()
            preds.append(pred)
    
    preds = np.concatenate(preds, axis=0)

    # Compute Inception Score using KL divergence and exponential
    split_scores = []
    N = preds.shape[0]
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits)]
        p_y = np.mean(part, axis=0)
        scores = [entropy(p_y, p) for p in part]
        split_scores.append(np.exp(np.mean(scores)))
    
    # Return mean and standard deviation of scores
    mean_score = np.mean(split_scores)
    std_score = np.std(split_scores)
    
    return mean_score, std_score

# Example usage:
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load example dataset (CIFAR-10) for testing
    dataset = datasets.ImageFolder(IMAGE_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    
    # Compute Inception Score
    mean, std = get_inception_score(images, batch_size=32, resize=True, splits=10)
    print(f"Inception Score: {mean:.3f} ± {std:.3f}")