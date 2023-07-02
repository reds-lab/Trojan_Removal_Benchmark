import numpy as np

class SIG():
    def __init__(self, size = 32, delta = 20, f = 6):
        self.size = size
        self.delta = delta
        self.f = f
        self.sig_trigger = self.plant_sin_trigger(shape = (self.size, self.size, 3), delta=self.delta, f=self.f)
        
    def plant_sin_trigger(self, shape = (32, 32, 3), delta=20, f=6):
        """
        Implement paper:
        > Barni, M., Kallas, K., & Tondi, B. (2019).
        > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
        > arXiv preprint arXiv:1902.11237
        superimposed sinusoidal backdoor signal with default parameters
        """
        alpha = 0.2
        pattern = np.zeros(shape)
        m = pattern.shape[1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

        return np.uint8((1 - alpha) * pattern)
    
    def img_poi(self, img):
        return img + self.sig_trigger

