
# Class for BadNets
class BadNets():
    def __init__(self, size = 32, position = 184):
        self.size = size
        self.position = position
    
    def img_poi(self, img):
        img[self.position:self.position+self.size,self.position:self.position+self.size,:] = 255
        return img

