import torch
import torch.nn as nn
from torch.optim import Adam
import sys

from models.discriminator import *
from models.unet import *
from util.config import *
from util.sampling import *
from training import *

if __name__ == "__main__":
    torch.manual_seed(42)
    log(f"Using {DEVICE}")
    mode = 'train'
    
    if sys.argv:
        mode = sys.argv[1] 
        print(mode)
    
    if mode == 'sample':
        model = UNet()
        model = load_model(model, f"model_diff_adv.pth")
    
        log('\nGenerating images...')
        
        generate_and_save('final1')
        generate_and_save('final2')
        generate_and_save('final3')
        generate_and_save('final4')
        generate_and_save('final5')
        
        log('\nDone.')
    
    elif mode == 'train': 
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
        train_diffusion_class_free(model, dataloader, optimizer, class_weights)
        
        log('\nSTEP 2')
        log('Training diffusion model with class embeddings...')
        train_diffusion_class(model, dataloader, optimizer, class_weights)
        
        log('\nSTEP 3')
        log('Training discriminator...')
        train_discriminator(discr, dataloader, optimizer_d, loss_d)
        
        log('\nSTEP 4')
        log('Finetuning...')
        finetune(model, discr, dataloader, optimizer, loss_d)
    else:
        log('Invalid command.')