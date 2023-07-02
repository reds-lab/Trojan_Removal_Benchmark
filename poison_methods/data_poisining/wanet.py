import copy
import torchvision
import torch
import torch.nn as nn

# Class for WaNet
class WaNet():
    def __init__(self, igrad_path, ngrid_path, s = 0.5, grid_rescale = 1):
        identity_grid = copy.deepcopy(torch.load(igrad_path))
        noise_grid = copy.deepcopy(torch.load(ngrid_path))
        h = identity_grid.shape[2]
        self.grid = identity_grid + s * noise_grid / h
        self.grid = torch.clamp(self.grid * grid_rescale, -1, 1)

    def img_poi(self, img):
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = torchvision.transforms.functional.convert_image_dtype(img, torch.float)
        poison_img = nn.functional.grid_sample(img.unsqueeze(0), self.grid, align_corners=True).squeeze()  # CHW
        img = poison_img.permute(1, 2, 0).numpy()
        # img = test_transform(img)
        return img