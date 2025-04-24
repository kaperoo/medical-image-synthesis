import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam

from models.discriminator import *
from models.unet import *
from util.config import *
from util.noise_scheduler import *
from util.helper_functions import *

# --- SAMPLING ---
@torch.no_grad()
def sample_timestep(x, class_label, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_output = model(x, t, class_label, CLASS_EMB_WEIGHT)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if torch.all(t == 0):
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def generate_and_save(idx=''):
    class_labels = torch.arange(NUM_CLASSES, dtype=torch.long, device=DEVICE)
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((NUM_CLASSES, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((NUM_CLASSES,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    
    filename = f"{FIG_PATH}{idx}.png"
    torchvision.utils.save_image(img, filename, normalize=True)
    log(f'Saved {filename}')

@torch.no_grad()
def generate_fake_images(batch_size, class_labels):
    img_size = (IMG_HEIGHT, IMG_WIDTH)
    img = torch.randn((batch_size, 1, img_size[0], img_size[1]), device=DEVICE)

    for i in range(0, TIMESTEPS)[::-1]:
        t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, class_labels, t)
        img = torch.clamp(img, -1.0, 1.0)
    return img 

# --- TRAINING ---
def train_diffusion_class_free(): 
    CLASS_EMB_WEIGHT = 0.0
    filename = f"{MODEL_PATH}_cf.pth"
    model.train()
    
    for epoch in range(CLASS_FREE_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
                
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, images, t, class_labels, class_weights)
            loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{CLASS_FREE_EPOCHS}, Loss: {loss.item()}")      

        if ((epoch + 1) % 10 == 0):
           generate_and_save(f'step1_epoch{epoch + 1}')
           torch.save(model.state_dict(), filename)
           log(f"Saved checkpoint.")
                
    torch.save(model.state_dict(), filename)
    log(f"Saved {filename}")

# --- TRAINING ---
def train_diffusion_class(): 
    CLASS_EMB_WEIGHT = 2.0
    filename = f"{MODEL_PATH}_ce.pth"
    model.train()
    
    for epoch in range(CLASS_EMB_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
                
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, images, t, class_labels, class_weights)
            loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{CLASS_EMB_EPOCHS}, Loss: {loss.item()}")      

        if ((epoch + 1) % 10 == 0):
           generate_and_save(f'step2_epoch{epoch + 1}')
           torch.save(model.state_dict(), filename)
           log(f"Saved checkpoint.")
                
    torch.save(model.state_dict(), filename)
    log(f"Saved {filename}")
    
def train_discriminator():
    filename = f'{MODEL_PATH}_discr.pth'
    discr.train()
    
    for epoch in range(DISCR_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer_d.zero_grad()
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)

            # Generate Fake Images
            if step % 50 == 0 or fake_images_cache is None:
                fake_images_cache = generate_fake_images(images.size(0), class_labels)
            fake_images = fake_images_cache.detach()

            # Compute discriminator loss
            optimizer_d.zero_grad()
                
            real_labels = torch.full((images.size(0), 1),  1.).to(DEVICE)
            fake_labels = torch.full((images.size(0), 1),  0.).to(DEVICE)

            real_preds = discr(images)
            real_loss = loss_d(real_preds[:images.size(0)], real_labels[:images.size(0)])

            fake_preds = discr(fake_images.detach())
            fake_loss = loss_d(fake_preds[:images.size(0)], fake_labels[:images.size(0)])

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
                
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{DISCR_EPOCHS}, Loss: {d_loss.item()}")    
            
    torch.save(discr.state_dict(), filename)
    log(f'Saved {filename}')
    
def finetune():
    filename_diff = f"{MODEL_PATH}.pth"
    model.train()
    discr.eval()
    
    for epoch in range(FINETUNE_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            #optimizer_d.zero_grad()
            
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (images.size(0),), device=DEVICE).long()
            
            if step % 50 == 0 or fake_images_cache is None:
                fake_images_cache = generate_fake_images(images.size(0), class_labels)
            fake_images = fake_images_cache.detach()
            
            # Train diffusion
            #diffusion_loss = get_loss(model, images, t, class_labels, class_weights)
            adv_loss = get_adversarial_loss(discr, fake_images.detach(), loss_d, DEVICE)
                
            #total_loss = diffusion_loss + ADV_LOSS_WEIGHT * adv_loss 
            adv_loss.backward()
            #total_loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{FINETUNE_EPOCHS}, Adv Loss: {adv_loss.item()},")    
                    
        if ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), filename_diff)
            
            log('Saved checkpoint.')
            generate_and_save(f'step3_epoch{epoch + 1}')

    torch.save(model.state_dict(), filename_diff)
    log(f'Saved models.')
    
if __name__ == "__main__":
    torch.manual_seed(42)
    log(f"Using {DEVICE}")
    
    dataloader = load_data()
    class_weights = get_class_weights(dataloader)
    
    model = UNet()
    discr = Discriminator()
    
    if torch.cuda.device_count() > 1:
        log(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
        discr = nn.DataParallel(discr)
        
    model.to(DEVICE)
    discr.to(DEVICE)    
        
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_d = Adam(discr.parameters(), lr=LEARNING_RATE_DISCR)
    
    loss_d = nn.BCELoss()
    
    log('\nSTEP 1')
    log('Training diffusion model class free...')
    train_diffusion_class_free()
    #model = load_model(model, f"{MODEL_PATH}_cf.pth")
    
    log('\nSTEP 2')
    log('Training diffusion model with class embeddings...')
    train_diffusion_class()
    #model = load_model(model,f"{MODEL_PATH}_ce.pth")
    
    log('\nSTEP 3')
    log('Training discriminator...')
    train_discriminator()
    #discr = load_model(discr, f"{MODEL_PATH}_discr.pth")
    
    log('\nSTEP 4')
    log('Finetuning...')
    finetune()
    
    #model = load_model(model, f"{MODEL_PATH}_ce.pth")
    #discr = load_model(discr, f"{MODEL_PATH}_discr.pth")
    
    log('\nGenerating images...')
    
    generate_and_save('final1')
    generate_and_save('final2')
    generate_and_save('final3')
    generate_and_save('final4')
    generate_and_save('final5')
    
    log('\nDone.')
