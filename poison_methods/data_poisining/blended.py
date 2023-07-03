import cv2
import numpy as np
import torch

# Class for Blended
class Blended():
    def __init__(self, noisy, img_size=224, clip_range = (0,255), mode='np'):
        self.noisy = noisy
        self.img_size = img_size
        self.clip_range = clip_range
        self.mode = mode
    
    def img_poi(self, img):
        if self.mode == 'np':
            if np.array(img).shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size)) + self.noisy
            else:
                img = img + self.noisy
            img = np.clip(img, self.clip_range[0],self.clip_range[1])
        else:
            img = torch.clip(img + self.noisy,self.clip_range[0],self.clip_range[1])
        return img

