import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

BATCH_SIZE = 8

FAKE_INPUT_PATH = "fake_data"
REAL_INPUT_PATH = "augmented_data"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# IMAGES MUST BE PREPROCESSED BEFOREHAND. MAKE SURE ALL IMAGES MATCH IN DIMENSIONS
# AND ENSURE THAT THE NO. OF IMAGES ARE THE SAME IN BOTH FOLDERS
real_dataset = datasets.ImageFolder(REAL_INPUT_PATH, transform=transform)
fake_dataset = datasets.ImageFolder(FAKE_INPUT_PATH, transform=transform)

## LOADING DATA AND COMPARE FOR ALL CLASSES
# TODO: INTRODUCE SEPARATE DATA LOADERS FOR EACH CLASS

fake_dataloader = DataLoader(fake_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

real_dataloader = DataLoader(real_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

## Below code refers to the original implementation of FID score calculation
# Implemented with the help of: https://minibatchai.com/2022/07/23/FID.html

def get_model():
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
    
    model.fc = torch.nn.Identity()
    return model

def get_moments(samples, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    with torch.no_grad():
        
        X_sum = torch.zeros((2048, 1)).to(device)
        XXT_sum = torch.zeros((2048, 2048)).to(device)
        count = 0

        for idx, (inp, _) in enumerate(samples):
            pred = model(inp.to(device))

            X_sum += pred.sum(dim=0, keepdim=True).T
            XXT_sum += (pred[:, None] * pred[..., None]).sum(0)
            count += len(inp)

    mean = X_sum / count
    cov = XXT_sum / count - mean @ mean.T

    return mean, cov

def frechet_inception_distance(m_w, C_w, m, C, debug=False):
    eigenvals = torch.linalg.eigvals(C @ C_w)
    trace_sqrt_CCw = eigenvals.real.clamp(min=0).sqrt().sum()
    if debug:
        print('Largest imaginary part magnitude:', eigenvals[eigenvals.imag > 0].abs().max().item())
        print('Most negative:', eigenvals[eigenvals.real < 0].real.min().item())
    fid = ((m - m_w)**2).sum() + C.trace() + C_w.trace() - 2 * trace_sqrt_CCw
    return fid

if __name__ == '__main__':
    model = get_model()

    m_real_data, C_real_data = get_moments(real_dataloader, model)
    m_fake_data, C_fake_data = get_moments(fake_dataloader, model)

    fid = frechet_inception_distance(m_real_data, C_real_data, m_fake_data, C_fake_data, debug=False)
    
    # The lower the score, the closer the generated images are to the real images
    print(fid)