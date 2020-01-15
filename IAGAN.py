import torch
import numpy as np
from PIL import Image

class Config():
    def __init__(self, 
                A, CSGM_iterations, IA_iterations, device, G_weights_dir, 
                z_dim, n_z_init, noise_lvl, lr_z_CSGM, lr_z_IA, lr_G, print_every):
        # The measurement operator (lambda function):
        self.A = A
        # Number of iterations for CSGM
        self.CSGM_iterations = CSGM_iterations
        # Number of image-adapted iterations
        self.IA_iterations = IA_iterations
        # The GPU to run with
        self.device = device
        # The generator 
        self.G_weights_dir = G_weights_dir
        # Size of the latent vector z
        self.z_dim = z_dim
        # Numebr of CSGM initialization 
        self.n_z_init = n_z_init
        # Noise level 
        self.noise_lvl = noise_lvl
        # Learning rates
        self.lr_z_CSGM = lr_z_CSGM
        self.lr_z_IA = lr_z_IA
        self.lr_G = lr_G
        # Print every
        self.print_every = print_every

def bicubic_kernel_2D(x, y, a=-0.5):
    # X
    abs_phase = np.abs(x)
    abs_phase3 = abs_phase**3
    abs_phase2 = abs_phase**2
    if abs_phase < 1:
        out_x = (a+2)*abs_phase3 - (a+3)*abs_phase2 + 1
    else:
        if abs_phase >= 1 and abs_phase < 2:
            out_x = a*abs_phase3 - 5*a*abs_phase2 + 8*a*abs_phase - 4*a 
        else:
            out_x = 0
    # Y
    abs_phase = np.abs(y)
    abs_phase3 = abs_phase**3
    abs_phase2 = abs_phase**2
    if abs_phase < 1:
        out_y = (a+2)*abs_phase3 - (a+3)*abs_phase2 + 1
    else:
        if abs_phase >= 1 and abs_phase < 2:
            out_y = a*abs_phase3 - 5*a*abs_phase2 + 8*a*abs_phase - 4*a 
        else:
            out_y = 0

    return out_x*out_y

def downsample_bicubic_2D(I, scale, device):
    # scale - integer > 1
    filter_supp = 4*scale + np.mod(scale, 2)
    is_even = 1 - np.mod(scale, 2)
    Filter = torch.zeros(1,1,filter_supp,filter_supp).float().to(device)
    grid = np.linspace(-(filter_supp//2) + 0.5*is_even, filter_supp//2 - 0.5*is_even, filter_supp)
    for n in range(filter_supp):
        for m in range(filter_supp):
            Filter[0, 0, m, n] = bicubic_kernel_2D(grid[n]/scale, grid[m]/scale)

    Filter = Filter/torch.sum(Filter)
    pad = np.int((filter_supp - scale)/2)
    I_padded = torch.nn.functional.pad(I, [pad, pad, pad, pad], mode='constant')
    I_out = torch.nn.functional.conv2d(I_padded, Filter, stride=(scale, scale))

    return I_out

def downsample_bicubic(I, scale, device):
    out = torch.zeros(I.shape[0], I.shape[1], I.shape[2]//scale, I.shape[3]//scale).to(device)
    for ch in range(out.shape[1]):
        out[:,ch:ch+1, :, :] = downsample_bicubic_2D(I[:, ch:ch+1, :, :], scale, device)
    return out

def tensorImg2npImg(I):
    return np.moveaxis(np.array(I[0,:].detach().cpu()), 0, 2)

def save_imag(I, dir):
    I = torch.clamp(I, 0, 1)
    I_np = tensorImg2npImg(I)
    if(I_np.shape[2] == 1):
        I_PIL = Image.fromarray(np.uint8(I_np[:,:,0]*255))
    else:
        I_PIL = Image.fromarray(np.uint8(I_np*255))
    I_PIL.save(dir)

def load_img(dir, device):
    I_PIL = Image.open(dir)
    I_np = np.array(I_PIL)/255.0
    if(np.shape(I_np.shape)[0] < 3):
        I = torch.tensor(I_np).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        I = torch.tensor(np.moveaxis(I_np, 2, 0)).float().unsqueeze(0).to(device)
    return I

def CSGM(y, G, config):
    G.load_state_dict(torch.load(config.G_weights_dir, map_location=config.device))
    G.eval()
    G.to(config.device)

    objective = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(config.z_dim)).to(config.device)

    csgm_losses = torch.zeros(config.n_z_init).to(config.device)
    # The following loop can be replaced by optimization in parallel, here it used 
    # due to memory limitation
    for i in range(config.n_z_init):
        if(config.n_z_init > 1):
            print('Z initialization number %d/%d' %(i+1, config.n_z_init))
        Z = torch.autograd.Variable(z_init[i:i+1, :], requires_grad = True)
        optimizer = torch.optim.Adam([{'params': Z, 'lr': config.lr_z_CSGM}])

        print('Running CSGM:')
        for step in range(config.CSGM_iterations):
            optimizer.zero_grad()
            Gz = G(Z)
            AGz = config.A(Gz)
            loss = objective(AGz, y)
            loss.backward()
            optimizer.step()
            if(step % config.print_every == 0):
                print('CSGM step %d/%d, objective = %.5f' %(step, config.CSGM_iterations, loss.item()))

        csgm_losses[i] = torch.sum( (y - AGz)**2 )
        z_init[i:i+1, :] = Z.detach()
    z_hat_idx = torch.argmin(csgm_losses)
    z_hat = z_init[z_hat_idx:z_hat_idx+1, :]
    I_CSGM = G(z_hat)

    return I_CSGM, z_hat

def IA(y, G, z_hat, config):
    G.load_state_dict(torch.load(config.G_weights_dir, map_location=config.device))
    G.eval()
    G.to(config.device)
    Z = torch.autograd.Variable(z_hat, requires_grad=True)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=config.lr_G)
    optimizer_z = torch.optim.Adam([{'params': Z, 'lr': config.lr_z_CSGM}])
    objective = torch.nn.MSELoss()

    print('Running image adaptive stage:')
    for step in range(config.IA_iterations):
        optimizer_z.zero_grad()
        optimizer_G.zero_grad()
        Gz = G(Z)
        AGz = config.A(Gz)
        loss = objective(AGz, y)
        loss.backward()
        optimizer_G.step()
        optimizer_z.step()
        if(step % config.print_every == 0):
            print('IA step %d/%d, objective = %.5f' %(step, config.IA_iterations, loss.item()))
    return Gz.detach()
    
def rand_mask(size, thresh):
    half_size = np.floor(size/2).astype('int32')
    idxX = np.mod(np.floor(np.abs(np.random.randn(half_size, half_size))*thresh), half_size)
    idxY = np.mod(np.floor(np.abs(np.random.randn(half_size, half_size))*thresh), half_size)
    mask_t = torch.zeros(size,size)
    mask_t[idxY, idxX] = 1
    # Duplicate
    dupIdx = [i for i in range(half_size-1, -1, -1)]
    mask_t[:half_size, half_size:] = mask_t[:half_size, dupIdx] # flip x
    mask_t[half_size:, :half_size] = mask_t[dupIdx, :half_size] # flip y
    x, y = np.meshgrid(dupIdx, dupIdx)
    mask_t[half_size:, half_size:] = mask_t[y, x] # flip x and y
    mask = np.array(mask_t)

    ratio = np.sum(mask==1)/mask.size
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).unsqueeze(4)
    mask_t = torch.cat((mask_t, mask_t), 4)
    return mask_t, ratio

def compress_FFT(x, mask):
    batch_size = x.shape[0]
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]
    R = torch.rfft(r, signal_ndim=2, normalized = True, onesided=False).float().view(batch_size, -1)
    G = torch.rfft(g, signal_ndim=2, normalized = True, onesided=False).float().view(batch_size, -1)
    B = torch.rfft(b, signal_ndim=2, normalized = True, onesided=False).float().view(batch_size, -1)
    mask = mask.view(-1)
    R_masked = R[:, mask == 1]
    G_masked = G[:, mask == 1]
    B_masked = B[:, mask == 1]
    X_masked = torch.cat((R_masked.unsqueeze(1), G_masked.unsqueeze(1), B_masked.unsqueeze(1)), dim = 1)
    return X_masked

def compress_FFT_t(X, mask):
    shape = mask.shape
    mask = mask.view(-1)
    R = torch.zeros_like(mask)
    R[mask == 1] = X[:, 0]
    R = R.reshape(shape)
    G = torch.zeros_like(mask)
    G[mask == 1] = X[:, 1]
    G = G.reshape(shape)
    B = torch.zeros_like(mask)
    B[mask == 1] = X[:, 2]
    B = B.reshape(shape)
    r = torch.irfft(R, signal_ndim=2, normalized = True, onesided=False)
    g = torch.irfft(G, signal_ndim=2, normalized = True, onesided=False)
    b = torch.irfft(B, signal_ndim=2, normalized = True, onesided=False)
    x = torch.cat((r, g, b), dim = 1)
    return x
