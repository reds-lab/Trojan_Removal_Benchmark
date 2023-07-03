import torch

import cv2
import numpy as np
import torch

# Class for Narcissus
class Narcissus():
    def __init__(self, noisy, clip_range = (-1,1), times=3):
        self.noisy = noisy
        self.clip_range = clip_range
        self.times = times
    
    def img_poi(self, img):
        return torch.clamp(img + self.noisy*self.times, self.clip_range[0], self.clip_range[1])

