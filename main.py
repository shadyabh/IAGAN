import torch
import cv2 as cv
import numpy as np

                #### Import the desired generator ####
from model import Generator

import os
import IAGAN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                #### Add your configuration here ####
### Measurement operator
## super resolution
A = lambda I: IAGAN.downsample_bicubic(I, 16, device)
A_dag = lambda I: torch.nn.functional.interpolate(I, scale_factor=16, mode='bicubic')
## compressed sensing
# mask, ratio = IAGAN.rand_mask(1024, 1024*0.5/2)
# print('FFT mask compression ratio = %f' %(ratio))
# A = lambda I: IAGAN.compress_FFT(I, mask)
# A_dag = lambda I: IAGAN.compress_FFT_t(I, mask)
## Noise level
noise_lvl = 0/255
## Number of iterations for CSGM
n_z_init = 5
## Latent vector size
z_dim = (n_z_init, 512, 1, 1)
## Images directory 
in_dir = './imgs/'
## Number of iterations
CSGM_itr = 1800
IA_itr = 300
## Generator
G = Generator()
G_saved_dir = './100_celeb_hq_network-snapshot-010403.pth'
G_out_range = (-1, 1)
## Learning rates
lr_z_CSGM = 0.1
lr_z_IA = 0.0001
lr_G = 0.001
## Print interval
print_interval = 10

config = IAGAN.Config(A, CSGM_itr, IA_itr, device, G_saved_dir, z_dim, 
                        n_z_init, noise_lvl, lr_z_CSGM, lr_z_IA, lr_G, print_interval)

IA_dir = os.path.join(in_dir, 'IAGAN/')
if(not os.path.isdir(IA_dir)):
    os.mkdir(IA_dir)
CSGM_dir = os.path.join(in_dir, 'CSGM/')
if(not os.path.isdir(CSGM_dir)):
    os.mkdir(CSGM_dir)

files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
GT_imgs = [f for f in files if ('.jpg' or '.png') in f]
GT_imgs.sort()

for img in GT_imgs:
    GT = IAGAN.load_img(in_dir + img, device)*(G_out_range[1] - G_out_range[0]) + G_out_range[0]
    y = A(GT)
    e = torch.randn_like(y)*config.noise_lvl
    y += e
    A_dagY = A_dag(y)

    # Run CSGM
    I_CSGM, z_hat = IAGAN.CSGM(y, G, config)
    IAGAN.save_imag((I_CSGM-G_out_range[0])/(G_out_range[1] - G_out_range[0]),
                     CSGM_dir + img[:-4] + '_CSGM.png')
    I_CSGM_BP = A_dagY + I_CSGM - A_dag(A(I_CSGM))
    IAGAN.save_imag((I_CSGM_BP-G_out_range[0])/(G_out_range[1] - G_out_range[0]),
                     CSGM_dir + img[:-4] + '_CSGM_BP.png')

    # Run IA
    I_IA = IAGAN.IA(y, G, z_hat, config)
    IAGAN.save_imag((I_IA-G_out_range[0])/(G_out_range[1] - G_out_range[0]),
                     IA_dir + img[:-4] + '_IA.png')
    I_IA_BP = A_dagY + I_IA - A_dag(A(I_IA))
    IAGAN.save_imag((I_IA_BP-G_out_range[0])/(G_out_range[1] - G_out_range[0]),
                     IA_dir + img[:-4] + '_IA_BP.png')








