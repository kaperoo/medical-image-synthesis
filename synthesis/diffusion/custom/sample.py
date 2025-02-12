import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from datetime import datetime
from unet import UNetConditional
from train import get_index_from_list, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, betas

# Set parameters
NUM_CLASSES = 10
T = 500  # Diffusion steps
IMG_SIZE = (28, 28)  # Adjust based on training size
OUTPUT_DIR = "generated"
TAG = "mnist_test"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image"""
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scale [-1,1] → [0,1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW → HWC
        transforms.Lambda(lambda t: t * 255.0),
        transforms.Lambda(lambda t: t.numpy().astype("uint8")),
    ])

    img = transform(tensor.squeeze(0))  # Remove batch dim
    return Image.fromarray(img.squeeze(), mode="L")  # Grayscale image

@torch.no_grad()
def sample_timestep(model, x, class_label, t, device):
    """Predicts noise and denoises the image step by step."""
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    predicted_noise = model(x, t, class_label)

    # Ensure shape consistency
    if predicted_noise.shape != x.shape:
        predicted_noise = torch.nn.functional.interpolate(predicted_noise, size=x.shape[2:], mode="bilinear", align_corners=False)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def generate_and_save_images(model, device="cuda"):
    """Generates and saves images for all classes."""
    model.eval()
    
    test_dir = os.path.join(OUTPUT_DIR, TAG)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_id in range(NUM_CLASSES):
        # Generate image from pure noise
        img = torch.randn((1, 1, *IMG_SIZE), device=device)
        class_tensor = torch.tensor([class_id], device=device)

        for i in reversed(range(T)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(model, img, class_tensor, t, device)
            img = torch.clamp(img, -1.0, 1.0)  # Keep within valid range

        # Convert and save image
        img_pil = tensor_to_image(img.cpu())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_pil.save(os.path.join(test_dir, f"generated_{timestamp}.png"))

        print(f"Saved image for class {class_id} in {test_dir}")

def load_model(model_path, num_classes=NUM_CLASSES, device="cuda"):
    """Loads trained model."""
    model = UNetConditional(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Load model
    model_path = "diffusion_custom.pth"
    model = load_model(model_path, NUM_CLASSES, device=device)

    # Generate images
    print("Generating images...")
    generate_and_save_images(model, device=device)
