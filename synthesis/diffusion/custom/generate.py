import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from unet import UNetConditional  # Import your U-Net

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetConditional()
model.load_state_dict(torch.load("diffusion_custom.pth", map_location=device))
model.to(device)
model.eval()

# Diffusion parameters (same as training)
T = 500  # Number of timesteps
betas = torch.linspace(0.0001, 0.02, T).to(device)  # Linear beta schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


# Function to reverse the diffusion process
@torch.no_grad()
def sample(model, img_size=(208, 560), class_label=0):
    """
    Generates an image by reversing the diffusion process.
    Args:
        model: Trained U-Net model
        img_size: Image size (height, width)
        class_label: Class index (0 to 6 for your 7-class model)
    Returns:
        Generated image tensor
    """
    x = torch.randn(1, 1, *img_size).to(device)  # Start from pure noise
    class_label = torch.tensor([class_label]).to(device)  # Class conditioning

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=device).float()  # Convert t to float
        predicted_noise = model(x, t_tensor, class_label)  # Predict noise

        if t > 0:
            noise = torch.randn_like(x)  # Sample random noise
            x = (
                sqrt_recip_alphas[t]
                * (x - betas[t] / sqrt_one_minus_alphas_cumprod[t] * predicted_noise)
                + torch.sqrt(betas[t]) * noise
            )
        else:
            x = sqrt_recip_alphas[t] * (
                x - betas[t] / sqrt_one_minus_alphas_cumprod[t] * predicted_noise
            )

    return x.squeeze().cpu().detach()  # Return final generated image


# Generate an image for each class
for class_id in range(7):  # Assuming 7 classes (0 to 6)
    img = sample(model, class_label=class_id)

    # Convert from [-1,1] to [0,1] for visualization
    img = (img + 1) / 2
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"Generated Image for Class {class_id}")
    plt.show()
