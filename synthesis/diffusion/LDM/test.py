# test the autoencoder
# from autoencoder import Autoencoder, Encoder, Decoder
from vae import Autoencoder, Encoder, Decoder
from torchvision import transforms
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the pretrained autoencoder
encoder = Encoder(
    channels=128,
    channel_multipliers=[1, 2, 4],
    n_resnet_blocks=2,
    in_channels=1,
    z_channels=4
)

decoder = Decoder(
    channels=128,
    channel_multipliers=[1, 2, 4],
    n_resnet_blocks=2,
    out_channels=1,
    z_channels=4
)

autoencoder = Autoencoder(
    encoder=encoder,
    decoder=decoder,
    emb_channels=4,
    z_channels=4
)
    
autoencoder.load_state_dict(torch.load("vae.pth"))
autoencoder.eval()

# Load an image
img_path = "../../../data/augmented_data/AMD/amd_1047099_1.jpg"

# scale the image to the correct size
data_transforms = [
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((128, 288)),
        transforms.Resize((128, 352)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        # transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        transforms.Normalize([0.5], [0.5]),
    ]
data_transform = transforms.Compose(data_transforms)
img = Image.open(img_path)

img = data_transform(img).unsqueeze(0)

# Pass the image through the autoencoder
# recon_img = autoencoder(img)
latent = autoencoder.encode(img).sample()
recon_img = autoencoder.decode(latent)

# latent = autoencoder.encoder(img)

# save latent to a txt file
# latent = latent[0].detach().cpu().numpy()
# np.savetxt("latent.txt", latent.reshape(-1, latent.shape[-1]))
# print(latent.shape)

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
plt.savefig("autoencoder_test.png")

plt.show()