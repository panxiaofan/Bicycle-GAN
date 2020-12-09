from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb
import wandb

##############################
#        Encoder 
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim = 8):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder. 

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
            
            Args in constructor: 
                latent_dim: latent dimension for z 
  
            Args in forward function: 
                img: image input (from domain B)

            Returns: 
                mu: mean of the latent code 
                logvar: sigma of the latent code 
        """

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)      
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Generator 
##############################
class Generator(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            latent_dim: latent dimension for z 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
    def __init__(self, latent_dim = 8, img_shape = (3,128,128)):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape
        self.latent_dim = latent_dim
        # (TODO: add layers...)
        # use U-Net Generator. See https://arxiv.org/abs/1505.04597
        self.downsample_1 = nn.Sequential(
            nn.Conv2d(in_channels = channels + latent_dim, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.downsample_3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.downsample_4 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.downsample_5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.downsample_6 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.downsample_7 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
        )
        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
        )
        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
        )
        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
        )
        self.upsample_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 512, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        self.upsample_6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.upsample_7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 3, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )
                                

    def forward(self, x, z):
        # (TODO: add layers...)
        # concatenate x and z
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), self.latent_dim, self.h, self.w)
        assert z.shape[0] == x.shape[0]
        x_concat = torch.cat([x, z], dim=1)
        
        down_1 = self.downsample_1(x_concat)
        down_2 = self.downsample_2(down_1)
        down_3 = self.downsample_3(down_2)
        down_4 = self.downsample_4(down_3)
        down_5 = self.downsample_5(down_4)
        down_6 = self.downsample_6(down_5)
        down_7 = self.downsample_7(down_6)

        up_1 = self.upsample_1(down_7)
        up_2 = self.upsample_2(torch.cat([up_1, down_6], dim=1))
        up_3 = self.upsample_3(torch.cat([up_2, down_5], dim=1))
        up_4 = self.upsample_4(torch.cat([up_3, down_4], dim=1))
        up_5 = self.upsample_5(torch.cat([up_4, down_3], dim=1))
        up_6 = self.upsample_6(torch.cat([up_5, down_2], dim=1))
        output = self.upsample_7(torch.cat([up_6, down_1], dim=1))
        
        return output 


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        """ The discriminator used in both cVAE-GAN and cLR-GAN
            
            Args in constructor: 
                in_channels: number of channel in image (default: 3 for RGB)

            Args in forward function: 
                x: image input (real_B, fake_B)
 
            Returns: 
                discriminator output: could be a single value or a matrix depending on the type of GAN
        """
        # Use PatchGAN discriminator. See https://arxiv.org/pdf/1611.07004 
        # We will have two discriminators with various output shape. The generator has to fool both of them. 
        
        # input shape - (N, 3, 128, 128)
        self.D_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 0, count_include_pad = False), # (N, 3, 64, 64)
            
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, stride = 2, padding = 1),  # (N, 32, 32, 32)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1), # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 1, padding = 1), # (N, 128, 15, 15)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 4, stride = 1, padding = 1), # (N, 1, 14, 14)
            nn.Sigmoid()
        )
        
        self.D_2 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),  # (N, 64, 64, 64)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1), # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 1, padding = 1), # (N, 256, 31, 31)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 4, stride = 1, padding = 1), # (N, 1, 30, 30)
            nn.Sigmoid()
        )

    def forward(self, x):
        d_out_1 = self.D_1(x)
        d_out_2 = self.D_2(x)

        return d_out_1, d_out_2







