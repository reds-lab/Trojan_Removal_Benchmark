
# Class for LC
class LC():
    def __init__(self, range = 255, alpha = 1):
        self.range = 255
        self.alpha = 1
    
    def img_poi(self, img):
        img[:3,:3] = 0
        img[:3,-3:] = 0
        img[-3:,:3] = 0
        img[-3:,-3:] = 0
        
        trigger_value = self.range*self.alpha

        img[-1, -1,:] = trigger_value
        img[-1, -3,:] = trigger_value
        img[-3, -1,:] = trigger_value
        img[-2, -2,:] = trigger_value
        img[0, -1,:] = trigger_value
        img[1, -2,:] = trigger_value
        img[2, -3,:] = trigger_value
        img[2, -1,:] = trigger_value
        img[0, 0,:] = trigger_value
        img[1, 1,:] = trigger_value
        img[2, 2,:] = trigger_value
        img[2, 0,:] = trigger_value
        img[-1, 0,:] = trigger_value
        img[-1, 2,:] = trigger_value
        img[-2, 1,:] = trigger_value
        img[-3, 0,:] = trigger_value
        
        return img

