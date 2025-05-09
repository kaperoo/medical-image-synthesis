from SWINGAN_model import *
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
from math import log2
import matplotlib.pylab as plt
import sys

sys.path.append(sys.argv[0])

vgg = models.vgg16(pretrained=True).features[:16].to(DEVICE).eval()
BATCH_SIZES = [16,32,32,16,16,16,16,8]
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

def perceptual_loss(real, fake):
    if real.shape[1] == 1:
        real = real.repeat(1, 3, 1, 1)
        fake = fake.repeat(1, 3, 1, 1)

    real_features = vgg(real)
    fake_features = vgg(fake)
    loss = F.mse_loss(real_features, fake_features)
    return loss

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

def generate_examples(gen, steps, loader=None, n=100, train=True):

    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIm).to(DEVICE)
            label = torch.randint(0, 6, (1, )).to(DEVICE)
            for image, _ in loader:
                idx = random.randint(0, image.size(0) - 1)
                random_image = image[idx]
                break
            random_image = random_image.unsqueeze(1)
            random_image = random_image.to(DEVICE)
            img = gen(noise, random_image, alpha, steps, label)
            savepath = os.path.join(SAVE_PATH, f"saved_examples/step{steps}")
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            savepathimage = os.path.join(savepath, f"img_{i}.png")
            save_image(img*0.5+0.5, savepathimage)

    if train:
        gen.train()

def check_loader():
    loader,_ = get_loader(128)
    cloth,_  = next(iter(loader))
    _,ax     = plt.subplots(3,3,figsize=(8,8))
    plt.suptitle('Some real samples')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ax[k][kk].imshow((cloth[ind].permute(1,2,0)+1)/2)
            ind +=1

def train_fn(
    critic,
    gen,
    loader,
    dataset,
    large_loader,
    large_dataset,
    step,
    alpha,
    opt_critic,
    opt_gen
):
    loop = tqdm(zip(loader, large_loader), leave=True)

    for batch_idx, ((real, label), (large_real, _)) in enumerate(loop):
        real = real.to(DEVICE)
        label = label.to(DEVICE)
        large_real = large_real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIm).to(DEVICE)

        fake  = gen(noise, large_real, alpha, step, label)
        critic_real = critic(real, alpha, step, label)
        critic_fake = critic(fake.detach(), alpha, step, label)
        gp = gradient_penalty(critic, real, fake, alpha, step, label, DEVICE)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001) * torch.mean(critic_real ** 2)
        )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step, label)
        adviserial_loss = -torch.mean(gen_fake)
        #loss_gen = -torch.mean(gen_fake)
        perc_loss = perceptual_loss(real, fake)
        loss_gen = adviserial_loss + WEIGHT*perc_loss #adjustment

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (
            PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset)
        )
        alpha = min(alpha,1)


        loop.set_postfix(
            gp = gp.item(),
            loss_critic = loss_critic.item(),
            perceptual_loss=perc_loss.item()
        )
    return alpha, loss_critic, perc_loss

if __name__ == "__main__":
    for param in vgg.parameters():
        param.requires_grad = False

    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if i == 0:
                continue
            # if arg == "--resume":
            #     save_path = os.path.join(SAVE_PATH, PATH_TO_CHECKPOINT)
            #     checkpoint = torch.load(os.path.join(save_path, "state.pth"))
            #     start_epoch = checkpoint['epoch']
            #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #     loss = checkpoint['loss']
            #     model.load_state_dict(torch.load(os.path.join(save_path, "model.pth")))
            #     # print(f"Resuming training from epoch {start_epoch}")
            elif arg == "--save":
                SAVE_PATH = sys.argv[i+1]
            elif arg == "--data":
                DATA_PATH = sys.argv[i+1]
            elif arg == "-l":
                LOG = True

    gen = Generator(
    Z_DIm, W_DIM, IN_CHANNELS, CHANNELS_IMG, 
    ).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
    opt_gen = optim.Adam([{'params': [param for name, param in gen.named_parameters() if 'map' not in name]}],
                        lr=LR, betas =(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr= LR, betas =(0.0, 0.99)
    )

    gen.train()
    critic.train()
    step = int(log2(START_TRAIN_IMG_SIZE / 4))

    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-7
        large_loader, large_dataset = get_loader(512, BATCH_SIZES[step])
        loader, dataset = get_loader(4*2**step)
        if LOG:
            log_path = os.path.join(SAVE_PATH, "log.txt")
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write('Curent image size: '+str(4*2**step)+'\n')
            else:
                with open(log_path, "a") as f:
                    f.write('Curent image size: '+str(4*2**step)+'\n')
        else:
            print('Curent image size: '+str(4*2**step))

        for epoch in range(num_epochs):
            alpha, loss, perc_loss = train_fn(
                critic, gen, loader, dataset, large_loader, large_dataset, step, alpha, opt_critic, opt_gen
            )
            # if epoch % 10 == 0:
            #     checkpoint = {
            #     'state_dict_G': gen.state_dict(),
            #     'state_dict_D': critic.state_dict(),
            #     'opt_G': opt_gen.state_dict(),
            #     'opt_D': opt_critic.state_dict(),
            #     'size_batch': num_epochs,
            #     'epoch': epoch,
            #     'step': step+1
            #     }
            #     torch.save(checkpoint, "checkpoint2.pth.tar")
            if LOG:
               with open(log_path, "a") as f:
                    f.write(f'Epoch [{epoch + 1}/ {num_epochs}, Loss: {loss.item()}, Perc_Loss: {perc_loss}\n') 
            else:
                print(f'Epoch [{epoch + 1}/ {num_epochs}, Loss: {loss.item()}, Perc_Loss: {perc_loss}\n')


        generate_examples(gen, step, large_loader)

        if step > 4:
            try:
                if not os.path.exists(os.path.join(SAVE_PATH, f'models')):
                    os.makedirs(os.path.join(SAVE_PATH, f'models'))
                torch.save(gen.state_dict(), os.path.join(SAVE_PATH, f'models/STYLEGAN_Discriminator{step}.pth'))
                torch.save(critic.state_dict(), os.path.join(SAVE_PATH, f'models/STYLEGAN_Generator{step}.pth'))
            except:
                pass
        
        step +=1