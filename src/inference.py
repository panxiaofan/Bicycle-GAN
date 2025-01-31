
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from datasets import *
from models import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import lpips
import torch
import os 
import numpy as np

try:
    os.makedirs('../inference')
    os.makedirs('../inference/real_images')
    os.makedirs('../inference/fake_images')
    os.makedirs('../inference/results')
    os.makedirs('../inference/eval_checkpoints')
except OSError as error:
    print('The folder has already existed!')


# Normalize image tensor
def norm(image):
    return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
    return ((tensor+1.0)/2.0)*255.0

def add_image_to_folder(iter, test_batch_size, real_B, fake_B):
    for i in range(test_batch_size):
        idx = iter * test_batch_size + i + 1
        plt.imsave('../inference/real_images/real_' + str(idx) + '.png', denorm(real_B[i]).type(torch.uint8).cpu().squeeze().permute(1,2,0).numpy())
        plt.imsave('../inference/fake_images/fake_' + str(idx) + '.png', denorm(fake_B[i]).type(torch.uint8).cpu().squeeze().permute(1,2,0).numpy())
    

# First create test data loader
test_batch_size = 2
test_img_dir = '../data/val/'
test_dataset = Edge2Shoe(test_img_dir)
test_size = len(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# indicate the device we will use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Load learnt Generator (final model)
img_shape = (3, 128, 128) 
latent_dim = 8
# checkpoint = torch.load('/Users/husiyun/Desktop/CIS 680/Final Project/bicyclegan_11_11999.pt', map_location=device)
checkpoint = torch.load('../checkpoint/bicyclegan_18_18999.pt', map_location=device)
generator = Generator(latent_dim, img_shape).to(device)
generator.load_state_dict(checkpoint['generator'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################
#        Quantitative Evaluation
##############################################

# preparation for LPIPS calculation
lpips_score = 0
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
        real_A = edge_tensor
        real_B = rgb_tensor
        random_z = Variable(Tensor(np.random.normal(0, 1, (test_batch_size,latent_dim))), requires_grad=False)
        fake_B = generator(real_A, random_z)
        # real_B_set.append(denorm(real_B.detach()))
        # fake_B_set.append(denorm(fake_B.detach()))

        # calculate LPIPS score
        lpips_score += torch.sum(loss_fn_alex(real_B, fake_B)).item()
        add_image_to_folder(idx, test_batch_size, real_B, fake_B)


print('LPIPS Score: {}'.format(lpips_score/test_size))

##############################################
#        Qualitative Evaluation
##############################################
with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
        real_A = edge_tensor
        real_B = rgb_tensor
        # sample three random_z from standard normal distribution
        random_z_1 = Variable(Tensor(np.random.normal(0, 1, (test_batch_size,latent_dim))), requires_grad=False)
        random_z_2 = Variable(Tensor(np.random.normal(0, 1, (test_batch_size,latent_dim))), requires_grad=False)
        random_z_3 = Variable(Tensor(np.random.normal(0, 1, (test_batch_size,latent_dim))), requires_grad=False)
        fake_B_1 = generator(real_A, random_z_1)
        fake_B_2 = generator(real_A, random_z_2)
        fake_B_3 = generator(real_A, random_z_3)

        # visualize the first image in the batch
        vis_real_B = denorm(real_A[0].detach()).cpu().data.numpy().astype(np.uint8)
        vis_fake_B_1 = denorm(fake_B_1[0].detach()).cpu().data.numpy().astype(np.uint8)
        vis_fake_B_2 = denorm(fake_B_2[0].detach()).cpu().data.numpy().astype(np.uint8)
        vis_fake_B_3 = denorm(fake_B_3[0].detach()).cpu().data.numpy().astype(np.uint8)

        fig, axs = plt.subplots(1,4, figsize = (10,10))
        axs[0].imshow(vis_real_B.transpose(1,2,0))
        axs[0].set_title('Real Images')
        axs[1].imshow(vis_fake_B_1.transpose(1,2,0))
        axs[1].set_title('Generated Images I')
        axs[2].imshow(vis_fake_B_2.transpose(1,2,0))
        axs[2].set_title('Generated Images II')
        axs[3].imshow(vis_fake_B_3.transpose(1,2,0))
        axs[3].set_title('Generated Images III')
        fig.savefig('../inference/results/res_{}.png'.format(idx*2+1))

        # visualize the second image in the batch
        vis_real_B = denorm(real_A[1].detach()).cpu().data.numpy().astype(np.uint8)
        vis_fake_B_1 = denorm(fake_B_1[1].detach()).cpu().data.numpy().astype(np.uint8)
        vis_fake_B_2 = denorm(fake_B_2[1].detach()).cpu().data.numpy().astype(np.uint8)
        vis_fake_B_3 = denorm(fake_B_3[1].detach()).cpu().data.numpy().astype(np.uint8)

        fig, axs = plt.subplots(1,4, figsize = (10,10))
        axs[0].imshow(vis_real_B.transpose(1,2,0))
        axs[0].set_title('Real Images')
        axs[1].imshow(vis_fake_B_1.transpose(1,2,0))
        axs[1].set_title('Generated Images I')
        axs[2].imshow(vis_fake_B_2.transpose(1,2,0))
        axs[2].set_title('Generated Images II')
        axs[3].imshow(vis_fake_B_3.transpose(1,2,0))
        axs[3].set_title('Generated Images III')
        fig.savefig('../inference/results/res_{}.png'.format(idx*2+2))


# ##############################################
# #        Visualize the Training
# ##############################################
fixed_z_1 = Variable(Tensor(np.random.normal(0, 1, (2,8))), requires_grad=False)
fixed_z_2 = Variable(Tensor(np.random.normal(0, 1, (2,8))), requires_grad=False)
fixed_z_3 = Variable(Tensor(np.random.normal(0, 1, (2,8))), requires_grad=False)

# for each checkpoint, we visualize the result of the first 10 images in the validation set
def eval_checkpoint(path, device, test_loader, fixed_z_1, fixed_z_2, fixed_z_3):
    # get the epoch from checkpoint path
    epoch = path.split('_')[-2]
    # load checkpoint
    img_shape = (3, 128, 128) 
    latent_dim = 8
    relative_path = '../checkpoint/'+path
    checkpoint = torch.load(relative_path, map_location=device)
    generator = Generator(latent_dim, img_shape).to(device)
    generator.load_state_dict(checkpoint['generator'])
    # visualize 
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 0):
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(device), norm(rgb_tensor).to(device)
            real_A = edge_tensor
            real_B = rgb_tensor
            fake_B_1 = generator(real_A, fixed_z_1)
            fake_B_2 = generator(real_A, fixed_z_2)
            fake_B_3 = generator(real_A, fixed_z_3)

            # visualize the first image in the batch
            vis_real_B = denorm(real_A[0].detach()).cpu().data.numpy().astype(np.uint8)
            vis_fake_B_1 = denorm(fake_B_1[0].detach()).cpu().data.numpy().astype(np.uint8)
            vis_fake_B_2 = denorm(fake_B_2[0].detach()).cpu().data.numpy().astype(np.uint8)
            vis_fake_B_3 = denorm(fake_B_3[0].detach()).cpu().data.numpy().astype(np.uint8)

            fig, axs = plt.subplots(1,4, figsize = (10,10))	
            axs[0].imshow(vis_real_B.transpose(1,2,0))
            axs[0].set_title('Real Images')
            axs[1].imshow(vis_fake_B_1.transpose(1,2,0))
            axs[1].set_title('Generated Images I')
            axs[2].imshow(vis_fake_B_2.transpose(1,2,0))
            axs[2].set_title('Generated Images II')
            axs[3].imshow(vis_fake_B_3.transpose(1,2,0))
            axs[3].set_title('Generated Images III')
            fig.savefig('../inference/eval_checkpoints/epoch_{}_res_{}.png'.format(epoch,idx*2+1))
            
            # visualize the second image in the batch
            vis_real_B = denorm(real_A[1].detach()).cpu().data.numpy().astype(np.uint8)
            vis_fake_B_1 = denorm(fake_B_1[1].detach()).cpu().data.numpy().astype(np.uint8)
            vis_fake_B_2 = denorm(fake_B_2[1].detach()).cpu().data.numpy().astype(np.uint8)
            vis_fake_B_3 = denorm(fake_B_3[1].detach()).cpu().data.numpy().astype(np.uint8)

            fig, axs = plt.subplots(1,4, figsize = (10,10))	
            axs[0].imshow(vis_real_B.transpose(1,2,0))
            axs[0].set_title('Real Images')
            axs[1].imshow(vis_fake_B_1.transpose(1,2,0))
            axs[1].set_title('Generated Images I')
            axs[2].imshow(vis_fake_B_2.transpose(1,2,0))
            axs[2].set_title('Generated Images II')
            axs[3].imshow(vis_fake_B_3.transpose(1,2,0))
            axs[3].set_title('Generated Images III')
            fig.savefig('../inference/eval_checkpoints/epoch_{}_res_{}.png'.format(epoch,idx*2+2))
            # plt.show()
            if idx > 3:
                break
            
# evaluate all checkpoints            
paths = os.listdir('../checkpoint')
for path in paths:          
    eval_checkpoint(path, device, test_loader, fixed_z_1, fixed_z_2, fixed_z_3)
            

        