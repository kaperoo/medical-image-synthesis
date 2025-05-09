from SWINGAN_model import *
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
from math import log2
import matplotlib.pylab as plt
import sys

sys.path.append(sys.argv[0])

vgg = models.vgg16(pretrained=True).features[:16].to(DEVICE).eval()
BATCH_SIZES = [32,32,32,16,16,16,16,8]
LOG = False
SAVE_PATH = "GAN_training/"
DATA_PATH = "GAN_training/data/augmented_data/"
START_TRAIN_IMG_SIZE = 4
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES)
LR = 1e-3
CHANNELS_IMG = 1
Z_DIm = 512
W_DIM = 512
IN_CHANNELS = 512
LAMBDA_GP = 10
WEIGHT = 0.01 # default is 0.01

labels = ["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"]
amounts = [0,500,500,500,500,500,500]

def get_loader(image_size, S=0):
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
         )
        ]
    )
    if S==0:
        batch_size = BATCH_SIZES[int(log2(image_size/4))]
    else:
        batch_size = S
    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return loader, dataset

def generate_examples(gen, steps, loader=None, n=100, train=True, label=0):

    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIm).to(DEVICE)
            label = torch.tensor([label]).to(DEVICE)
            for image, _ in loader:
                idx = random.randint(0, image.size(0) - 1)
                random_image = image[idx]
                break
            random_image = random_image.unsqueeze(1)
            random_image = random_image.to(DEVICE)
            img = gen(noise, random_image, alpha, steps, label)
            savepath = os.path.join(SAVE_PATH, f"2layer/{labels[label]}")
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savepathimage = os.path.join(savepath, f"img_{i}.png")
            save_image(img*0.5+0.5, savepathimage)

    if train:
        gen.train()

gen = Generator(
Z_DIm, W_DIM, IN_CHANNELS, CHANNELS_IMG, 
).to(DEVICE)

gen.load_state_dict(torch.load("GAN_training/50_epoch_final_model_2layer/models/STYLEGAN_Discriminator7.pth"))
gen.eval()

loader, _ = get_loader(512)

for i in range(len(labels)):
    generate_examples(gen, 7, loader=loader, n=amounts[i], train=False, label=i)


