import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import torchvision.utils as vutils

import matplotlib.pyplot as plt

from generator import Generator_CNN
from discriminator import Discriminator_CNN

class GAN(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.generator = Generator_CNN(device=self.device, dtype=self.dtype)
        self.discriminator = Discriminator_CNN(device=self.device, dtype=self.dtype)
    
    def train_gan(self,
              device,
              dtype,
              real_data,
              lrg,
              lrd,
              betas,
              num_epochs,
              max_grad_norm):
        optimiser_G = optim.Adam(self.generator.parameters(), lr=lrg, betas=betas)
        optimiser_D = optim.Adam(self.discriminator.parameters(), lr=lrd, betas=betas)
        criterion = nn.BCELoss()
        gen_loss = []
        disc_loss = []
        count = 0
        
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            avg_gen_loss = 0
            avg_disc_loss = 0
            for real_batch in real_data:
                real_batch = real_batch #+ 0.15 * torch.randn_like(real_batch) ##noise addition to image
                B, C, H, W = real_batch.shape
                
                Z = torch.randn((B, 100), device=device, dtype=dtype)
                fake_batch = self.generator(Z)
    
                fake_labels = torch.ones(B, 1, device=device, dtype=dtype)
                real_labels = torch.zeros(B, 1, device=device, dtype=dtype)
                
                smooth_fake_labels = fake_labels - 0.1 * torch.rand(B, 1, device=device, dtype=dtype)
                smooth_real_labels = real_labels + 0.1 * torch.rand(B, 1, device=device, dtype=dtype)
                
                ## DISCRIMINATOR TRAINING
                optimiser_D.zero_grad()
                loss_d = 0
                
                real_discriminator_out = self.discriminator(real_batch)
                fake_discriminator_out = self.discriminator(fake_batch.detach())
                
                loss_d_real = criterion(real_discriminator_out, smooth_real_labels)
                loss_d_fake = criterion(fake_discriminator_out, smooth_fake_labels)
                loss_d += (loss_d_real + loss_d_fake) / 2
                    
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_grad_norm)
                optimiser_D.step()
                    
                disc_loss.append(loss_d.item())
                avg_disc_loss += loss_d.item()
                
                
                ## GENERATOR TRAINING
                optimiser_G.zero_grad()
                loss_g = 0
                
                fake_discriminator_out = self.discriminator(fake_batch)
                loss_g += criterion(fake_discriminator_out, smooth_real_labels)
                loss_g.backward() 
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_grad_norm)                    
                optimiser_G.step()
                    
                gen_loss.append(loss_g.item())
                avg_gen_loss += loss_g.item()
                
                
            count += 1
            print(f'Epoch: {count}, Generator Loss: {avg_gen_loss/len(real_data):.4f}, Discriminator Loss: {avg_disc_loss/len(real_data):.4f}')
            
            if epoch % 1 == 0 or epoch == num_epochs - 1:
                self.monitor_generated_images(epoch, Z[:16], real_batch)  
            
        return gen_loss, disc_loss
    
            
    def monitor_generated_images(self, epoch, fixed_noise, real_batch, num_images=16, figsize=(20,10)):
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(fixed_noise).detach().cpu()
            real_images = real_batch[:num_images].cpu()
            
            fake_grid = vutils.make_grid(fake_images[:num_images], padding=2, normalize=True)
            real_grid = vutils.make_grid(real_images, padding=2, normalize=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            ax1.axis("off")
            ax1.set_title("Real Images")
            ax1.imshow(real_grid.permute(1, 2, 0).mean(dim=2), cmap='viridis')
            
            ax2.axis("off")
            ax2.set_title(f"Generated Images (Epoch {epoch+1})")
            ax2.imshow(fake_grid.permute(1, 2, 0).mean(dim=2), cmap='viridis') 

            plt.savefig(f'/home/mtech1/gan/images/test/comparison_epoch_{epoch+1}.png')
            plt.close(fig)


