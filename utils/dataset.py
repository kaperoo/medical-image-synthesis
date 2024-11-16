import os
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


def show_images(datset, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(torchvision.datasets.ImageFolder(datset)):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0])

    plt.show()


def check_data_sizes(data):
    sizes = []
    for folder in os.listdir(data):
        folder_path = os.path.join(data, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                img = Image.open(file_path)
                sizes.append(img.size)

    # scatter plot of image sizes
    plt.scatter(*zip(*sizes), alpha=0.5)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Sizes")
    plt.show()
