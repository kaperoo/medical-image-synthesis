import torch
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from diffusion_model_cond_hires import (
    SimpleUnet,
    get_index_from_list,
    sqrt_recip_alphas,
    posterior_variance,
    sqrt_one_minus_alphas_cumprod,
    betas,
)


def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image"""
    # Reverse the normalization and channel ordering
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),  # [-1,1] -> [0,1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW -> HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype("uint8")),
        ]
    )

    # Convert to PIL Image
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image if it's a batch
    img = reverse_transforms(tensor)
    return Image.fromarray(img.squeeze(), mode="L")  # squeeze for grayscale


@torch.no_grad()
def sample_timestep(model, x, class_label, t, device):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, class_label) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def generate_and_save_images(
    model,
    output_dir="generated_images",
    num_images=1,
    class_label=0,
    device="cuda",
    T=1000,
    img_size=(560, 208),
):
    """
    Generate images using the trained diffusion model and save them to files

    Args:
        model: The trained diffusion model
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        class_label: Class label for conditional generation
        device: Device to run generation on
        T: Number of timesteps in the diffusion process
        img_size: Size of the images to generate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model.eval()

    # Sample noise
    img = torch.randn((num_images, 1, img_size[0], img_size[1]), device=device)

    # Set class label tensor
    class_labels = torch.tensor([class_label] * num_images, device=device)

    # Sampling loop
    for i in range(T)[::-1]:
        t = torch.full((num_images,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, class_labels, t, device)
        img = torch.clamp(img, -1.0, 1.0)

    # Save each generated image
    for idx in range(num_images):
        # Convert tensor to PIL Image
        pil_image = tensor_to_image(img[idx].cpu())

        # Create filename with timestamp, class label, and index
        filename = f"generated_{timestamp}_class{class_label}_img{idx}.png"
        filepath = os.path.join(output_dir, filename)

        # Save the image
        pil_image.save(filepath)
        print(f"Saved image to {filepath}")

    return img


def load_model(model_path, num_classes=7, device="cuda"):
    """Load a saved diffusion model"""
    model = SimpleUnet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Example usage
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_path = "diffusion_model_cond_hires.pth"
    model = load_model(model_path, device=device)

    # Generate and save images
    print("Generating images...")

    # # Generate 4 images of class 0
    # generated_images = generate_and_save_images(
    #     model,
    #     output_dir="generated_images",
    #     num_images=4,
    #     class_label=0,
    #     device=device,
    #     img_size=(546, 200),
    # )

    # # Generate 4 images of class 1
    # generated_images_class1 = generate_and_save_images(
    #     model,
    #     output_dir="generated_images",
    #     num_images=4,
    #     class_label=1,
    #     device=device,
    #     img_size=(546, 200),
    # )

    # Generate an image for each class
    for i in range(7):
        generate_and_save_images(
            model,
            output_dir="generated_images",
            num_images=1,
            class_label=i,
            device=device,
            img_size=(208, 560),
        )
