import torch

def Narcissus(img, noisy, times=3):
    return torch.clip(img + noisy*times,-1,1)