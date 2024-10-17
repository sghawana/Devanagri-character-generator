import torch
from torch.utils.data import DataLoader

from trainer import GAN
from data import devnagari, load_images_from_directory

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


### Loading Dataset

ROOT_DIRECTORY = '/home/mtech1/gan/data/all'  
BATCH_SIZE = 128

images = load_images_from_directory(ROOT_DIRECTORY)
print(f"Loaded {len(images)} images.")

real_dataset = devnagari(images, DEVICE, DTYPE)
real_dataloader = DataLoader(dataset=real_dataset,
                              batch_size=256, 
                              shuffle=True
                              )

### Training

NUM_EPOCHS = 10 

BETAS = (0.5, 0.999) #Adam optimiser(b1,b2)
LEARNING_RATE_DISCRIMINATOR = 0.0001
LEARNING_RATE_GENERATOR = 0.0002

LEARNING_RATE = 0.0002
MAX_GRAD_NORM = 1


training_params={'device': DEVICE,
                 'dtype': DTYPE,
                 'real_data': real_dataloader,
                 'lrg': LEARNING_RATE_GENERATOR,
                 'lrd': LEARNING_RATE_DISCRIMINATOR,
                 'betas': BETAS,
                 'num_epochs': NUM_EPOCHS,
                 'max_grad_norm': MAX_GRAD_NORM}


my_gan = GAN(device=DEVICE, dtype=DTYPE)
gen_loss, disc_loss = my_gan.train_gan(**training_params)