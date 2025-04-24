import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

from util.config import *
from util.noise_scheduler import *

def log(string):
    if not OUTPUT_TO_FILE:
        print(string)
    else:
        file = open(LOGS_PATH, "a")
        file.write(string)
        file.write('\n')
        file.close()

def get_loss(model, x_0, t, class_labels, class_weights, class_emb_weight):
    loss = nn.SmoothL1Loss(beta=0.1) 
    
    t = t[:x_0.shape[0]]
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t, class_labels, class_emb_weight)
    
    per_sample_loss = loss(noise, noise_pred)
    weights = class_weights[class_labels].view(-1,1,1,1)
    weighted_loss = (per_sample_loss * weights).mean()

    return weighted_loss

def get_adversarial_loss(discr, fake_images, loss_d, device=DEVICE):
    fake_preds = discr(fake_images)
    real_labels = torch.full_like(fake_preds, 0.9, device=device)
    return loss_d(fake_preds, real_labels)

def load_data(): 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_HEIGHT_SCALED, IMG_WIDTH)),
        transforms.Pad((0, PADDING)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = datasets.ImageFolder(PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    return dataloader

def get_class_weights (dataloader):
    labels = np.concatenate([batch[1].numpy() for batch in dataloader])
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    return class_weights
    
def load_model(model, path):
    model.to(DEVICE)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
    return model

def reverse_image (image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    return reverse_transforms(image)