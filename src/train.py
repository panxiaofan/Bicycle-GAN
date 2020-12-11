import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from vis_tools import *
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import time
import pdb
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Normalize image tensor
def norm(image):
    return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
    return ((tensor+1.0)/2.0)*255.0

# Reparameterization helper function 
# (You may need this helper function here or inside models.py, depending on your encoder implementation)

def reparameterization(mu, logvar):    
    std = torch.exp(logvar/2)
    eps = torch.randn_like(std)
    z = mu + std*eps
    return z


##############################
#        Loss
##############################

# calculate the L-1 loss between transformed image G(A, z) and target image B
def loss_generator(G, real_img, z, target_img, criterion_l1):
    '''
    Input:
    G - generator
    real_img - the original image, A
    z - encoded z, the output of raparameterization
    target_img - the target image, B
    
    Output:
    L1 loss between image generated by generator and our target image B
    '''
    # generating image
    fake_img = G(real_img, z)
    assert fake_img.shape == target_img.shape
    return criterion_l1(fake_img, target_img)

# calculate the adversarial loss
def loss_discriminator(fake_img, D, real_img, criterion_bce, Tensor):
    '''
    This function could be used for the adversarial loss in both cVAE-GAN and cLR-GAN
    Input:
    fake_img - fake images generate by the generator. 
    D - Discriminator
    real_img: original image A
    valid: label of valid output, which equals to 1
    fake: label of invalid output, which equals to 0
    
    Output:
    - loss_D: adversarial loss
    '''
    # since we use PatchGan Discriminator, we return two outputs
    real_d_out_1,  real_d_out_2 = D(real_img)
    valid_1 = Variable(Tensor(np.ones(real_d_out_1.shape)), requires_grad=False)
    valid_2 = Variable(Tensor(np.ones(real_d_out_2.shape)), requires_grad=False)
    real_loss_1 = criterion_bce(real_d_out_1, valid_1)
    real_loss_2 = criterion_bce(real_d_out_2, valid_2)

    fake_d_out_1, fake_d_out_2 = D(fake_img)
    fake_1 = Variable(Tensor(np.zeros(fake_d_out_1.shape)), requires_grad=False)
    fake_2 = Variable(Tensor(np.zeros(fake_d_out_2.shape)), requires_grad=False)

    fake_loss_1 = criterion_bce(fake_d_out_1, fake_1)
    fake_loss_2 = criterion_bce(fake_d_out_2, fake_2)

    # loss_D = (real_loss_1 + real_loss_2)/2 + (fake_loss_1 + fake_loss_2)/2
    loss_D = real_loss_1 + fake_loss_1 + real_loss_2 +  fake_loss_2

    return loss_D

def loss_discriminator_fix_D(fake_img, D, criterion_mse, Tensor):
    fake_d_1, fake_d_2 = D(fake_img)
    valid1 = Variable(Tensor(np.ones(fake_d_1.shape)), requires_grad=False)
    valid2 = Variable(Tensor(np.ones(fake_d_2.shape)), requires_grad=False)
    fake_loss_1 = criterion_mse(fake_d_1, valid1)
    fake_loss_2 = criterion_mse(fake_d_2, valid2)
    loss_vae_gan = fake_loss_1 + fake_loss_2
    return loss_vae_gan




# compute the KL-Divergence between N(0,1) and latent space
def kld(mu, logvar):
    '''
    Input:
    mu - output of the Encoder, the mean of normal distribution
    logvar - output of the Encoder, the log value of variance
    
    Output:
    KL-divengence
    '''
    
    return torch.sum(0.5*(mu ** 2 + torch.exp(logvar) - logvar - 1))
    

def z_loss(fake_img, encoder, random_z, criterion_l1):
    '''
    Input:
    fake_img - fake images generated by the generator, B_hat
    encoder - the trained Encoder
    random_z - randomly generated standard normal distribution, the prior distribution of p(z)
    
    Output:
    L1 los between the encoded latent vector of B_hat and prior distribution p(z)
    '''
    mu, logvar = encoder(fake_img.detach())
    z_loss = criterion_l1(mu, random_z)
    return z_loss

def all_zero_grad(optimizer1, optimizer2, optimizer3, optimizer4):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
   

############################# reproductivity #############################
torch.manual_seed(1);
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
############################# Training Configurations #############################
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
num_epochs = 20
batch_size = 48
lr_rate = 0.0002  	      # Adam optimizer learning rate
beta1 = 0.5			  # Adam optimizer beta 1, beta 2
beta2 = 0.999

lambda_pixel =  10      # Loss weights for pixel loss
lambda_latent =  0.5     # Loss weights for latent regression
lambda_kl =  0.01         # Loss weights for kl divergence
latent_dim =  8        # latent dimension for the encoded images from domain B
report_freq = 10      #visualize image every 'report_freq' iters
visual_freq = 1000
save_freq = 1000      #save models every 'save_freq' iters
############################# Training datasets #############################
# img_dir = '/home/zlz/BicycleGAN/datasets/edges2shoes/train/'
img_dir = '../data/train/'
dataset = Edge2Shoe(img_dir)
loader = data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#save models for retrain purpose
os.makedirs("../checkpoint", exist_ok = True)

# tensorboard
os.makedirs("../logs", exist_ok = True)
writer = SummaryWriter(log_dir="../logs")

# Loss functions
criterion_l1 = torch.nn.L1Loss().to(device)
criterion_mse = torch.nn.MSELoss().to(device)

# Define generator, encoder and discriminators
encoder = Encoder(latent_dim).to(device)
generator = Generator(latent_dim, img_shape).to(device)
D_VAE = Discriminator().to(device)
D_LR = Discriminator().to(device)

# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=(beta1,beta2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(beta1,beta2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=(beta1,beta2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=(beta1,beta2))

# For adversarial loss (optional to use)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
torch.autograd.set_detect_anomaly(True)
# losss recorder
running_loss_l1_image = 0
running_loss_gan_vae = 0
running_loss_kl = 0
running_loss_l1_z = 0
running_loss_gan = 0
running_total_loss = 0

loss_l1_image_list = []
loss_gan_vae_list = []
loss_kl_list = []
loss_l1_z_list = []
loss_gan_list = []
total_loss_list = []
# Training
total_steps = len(loader)*num_epochs; step = 0

img_num = 36

# fixed_rand_z = torch.randn(36, latent_dim, batch_size, 1, dtype=Tensor)
fixed_rand_z = Variable(Tensor(np.random.normal(0, 1, (batch_size, img_num, latent_dim))), requires_grad=False)

for e in range(num_epochs):
    generator.train()
    encoder.train()
    D_LR.train()
    D_VAE.train()
    # start = time.time()
    for idx, data in enumerate(loader):
        ########## Process Inputs ##########
        edge_tensor, rgb_tensor = data
        # edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id
        edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
        real_A = edge_tensor
        real_B = rgb_tensor
        #----------------------------------
        #  Train Discriminator (cVAE-GAN)
        #----------------------------------
        loss_D = 0
        # generate loss in cVAE-GAN
        mu, logvar = encoder(real_B)
        z = reparameterization(mu, logvar)
        fake_B_cVAE = generator(real_A, z)
        loss_d_gan_vae = loss_discriminator(fake_B_cVAE, D_VAE, real_B, criterion_mse, Tensor)
        # generate loss in cLR-GAN
        random_z = Variable(Tensor(np.random.normal(0, 1, (batch_size,latent_dim))), requires_grad=False)
        fake_B_cLR = generator(real_A, random_z)
        loss_d_gan = loss_discriminator(fake_B_cLR, D_LR, real_B, criterion_mse, Tensor)
        loss_D = loss_d_gan_vae + loss_d_gan
        #update
        all_zero_grad(optimizer_G, optimizer_E, optimizer_D_VAE, optimizer_D_LR)
        loss_D.backward()
        optimizer_D_VAE.step()
        optimizer_D_LR.step()
        #-------------------------------
        #  Train Generator and Encoder
        #------------------------------
        loss_EG = 0
        loss_G = 0
        # GAN loss in cVAE-GAN
        mu, logvar = encoder(real_B)
        z = reparameterization(mu, logvar)
        fake_B_cVAE = generator(real_A, z)
        loss_G_cVAE = loss_discriminator_fix_D(fake_B_cVAE, D_VAE, criterion_mse, Tensor)
        # GAN loss in cLR-GAN
        random_z = Variable(Tensor(np.random.normal(0, 1, (batch_size,latent_dim))), requires_grad=False)
        fake_B_cLR = generator(real_A, random_z)
        loss_G_cLR = loss_discriminator_fix_D(fake_B_cLR, D_LR, criterion_mse, Tensor)
        loss_G_gan_vae = loss_G_cVAE + loss_G_cLR
        # KL-divergence loss in cVAE-GAN
        loss_kl = kld(mu, logvar)
        KL_div = lambda_kl * loss_kl
        # Reconstruction loss in cVAE-GAN
        l1_image = loss_generator(generator, real_A, z, real_B, criterion_l1)
        loss_l1_image = lambda_pixel * l1_image
        loss_EG = loss_G_gan_vae + KL_div + loss_l1_image
        all_zero_grad(optimizer_G, optimizer_E, optimizer_D_VAE, optimizer_D_LR)
        loss_EG.backward(retain_graph=True)
        optimizer_G.step()
        optimizer_E.step()
        #-------------------------------
        #  Train only Generator
        #------------------------------

        loss_l1_z = z_loss(fake_B_cLR, encoder, random_z, criterion_l1)
        loss_G = lambda_latent * loss_l1_z
        all_zero_grad(optimizer_G, optimizer_E, optimizer_D_VAE, optimizer_D_LR)
        loss_G.backward()
        optimizer_G.step()

        """ Optional TODO: 
            1. You may want to visualize results during training for debugging purpose
            2. Save your model every few iterations
        """
        running_total_loss += (loss_D + loss_EG + loss_G).item()
        running_loss_l1_image += l1_image.item()
        running_loss_gan_vae += (loss_d_gan_vae + loss_G_cVAE).item()
        running_loss_gan += (loss_d_gan + loss_G_cLR).item()
        running_loss_kl += loss_kl.item()
        running_loss_l1_z += loss_l1_z.item()

        ##################### Visualization #########################################
        if step % report_freq == report_freq - 1:
            print(
                'Train Epoch: {} {:.0f}% \tTotal Loss: {:.6f} \tLoss_l1_image: {:.6f}\tLoss_VAE_GAN: {:.6f}\tLoss_KL: {:.6f}\tLoss_l1_latent: {:.6f}\tLoss_GAN: {:.6f}'.format
                (e + 1, 100. * idx / len(loader), running_total_loss / report_freq,
                 running_loss_l1_image / report_freq, running_loss_gan_vae / report_freq,
                 running_loss_kl / report_freq, running_loss_l1_z / report_freq,
                 running_loss_gan / report_freq))

            ##################### write to summary writer #########################################
            writer.add_scalar('Loss/train/total_loss', running_total_loss, len(total_loss_list))
            writer.add_scalar('Loss/train/loss_l1_image', running_loss_l1_image, len(loss_l1_image_list))
            writer.add_scalar('Loss/train/loss_gan_vae', running_loss_gan_vae, len(loss_gan_vae_list))
            writer.add_scalar('Loss/train/loss_gan', running_loss_gan, len(loss_gan_list))
            writer.add_scalar('Loss/train/loss_kl', running_loss_kl, len(loss_kl_list))
            writer.add_scalar('Loss/train/loss_l1_z', running_loss_l1_z, len(loss_l1_z_list))

            # record loss
            total_loss_list.append(running_total_loss)
            loss_l1_image_list.append(running_loss_l1_image)
            loss_gan_vae_list.append(running_loss_gan_vae)
            loss_gan_list.append(running_loss_gan)
            loss_kl_list.append(running_loss_kl)
            loss_l1_z_list.append(running_loss_l1_z)

            # reset
            running_loss_l1_image = 0
            running_loss_gan_vae = 0
            running_loss_kl = 0
            running_loss_l1_z = 0
            running_loss_gan = 0
            running_total_loss = 0
            # end = time.time()
            # print(e, step, 'T: ', end - start)
            # start = end
        ########## Save Generators ##########
        if step % save_freq == save_freq - 1:
            checkpoint = {
                    'epoch':e,
                    'step':step,
                    'generator':generator.state_dict(),
                    'encoder':encoder.state_dict(),
                    'D_VAE':D_VAE.state_dict(),
                    'D_LR':D_LR.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_E': optimizer_E.state_dict(),
                    'optimizer_D_VAE':optimizer_D_VAE.state_dict(),
                    'optimizer_D_LR':optimizer_D_LR.state_dict()
                          }
            torch.save(checkpoint, '../checkpoint/bicyclegan_{}_{}.pt'.format(e,step))
        step += 1
    writer.close()





