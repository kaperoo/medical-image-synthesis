# test the autoencoder
from autoencoder import Autoencoder, Encoder
from torchvision import transforms
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the pretrained autoencoder
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pth"))
autoencoder.eval()

# Load an image
img_path = "../../../data/augmented_data/AMD/amd_1047099_1.jpg"

# scale the image to the correct size
data_transforms = [
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((128, 288)),
        transforms.Resize((128, 352)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
data_transform = transforms.Compose(data_transforms)
img = Image.open(img_path)

img = data_transform(img).unsqueeze(0)

# Pass the image through the autoencoder
recon_img = autoencoder(img)

# Display the original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(recon_img[0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')


# #save recon_img with pillow
# reverse_transforms = transforms.Compose([
#         transforms.Lambda(lambda t: (t + 1) / 2),
#         transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
#         transforms.Lambda(lambda t: t * 255.),  # Scale to [0, 255]
#         transforms.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),  # Convert to uint8
#         transforms.ToPILImage(),
#     ])

# recon_img_pil = reverse_transforms(recon_img[0])
# recon_img_pil.save("recon_img.jpg")

# save the image
# plt.savefig("autoencoder_test.png")

plt.show()