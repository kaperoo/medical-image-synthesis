import torch

from models.discriminator import *
from models.unet import *
from util.config import *
from util.noise_scheduler import *
from util.helper_functions import *
from util.sampling import *

# --- TRAINING ---
def train_diffusion_class_free(model, dataloader, optimizer, class_weights): 
    class_emb_weight = 0.0
    filename = f"{MODEL_PATH}_cf.pth"
    model.train()
    
    for epoch in range(CLASS_FREE_EPOCHS):
        for _, batch in enumerate(dataloader):
            optimizer.zero_grad()
                
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, images, t, class_labels, class_weights, class_emb_weight)
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
def train_diffusion_class(model, dataloader, optimizer, class_weights): 
    class_emb_weight = 2.0
    filename = f"{MODEL_PATH}_ce.pth"
    model.train()
    
    for epoch in range(CLASS_EMB_EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
                
            images, class_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                
            t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, images, t, class_labels, class_weights, class_emb_weight)
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
    
def train_discriminator(discr, dataloader, optimizer_d, loss_d):
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
    
def finetune(model, discr, dataloader, optimizer, loss_d):
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
            adv_loss = get_adversarial_loss(discr, fake_images.detach(), loss_d, DEVICE)
            adv_loss.backward()
            optimizer.step()
                
        if (OUTPUT_TO_FILE):
            log(f"Epoch {epoch + 1}/{FINETUNE_EPOCHS}, Adv Loss: {adv_loss.item()},")    
                    
        if ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), filename_diff)
            
            log('Saved checkpoint.')
            generate_and_save(f'step3_epoch{epoch + 1}')

    torch.save(model.state_dict(), filename_diff)
    log(f'Saved models.')