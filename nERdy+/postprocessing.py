from skimage import restoration
from skimage.filters import threshold_otsu
import copy

class PostProcessing:

    def __init__(self):
        pass

    def postprocessing(self, img):
        # get background
        img_bg = restoration.rolling_ball(img)

        # get foreground
        img_fg = self - img_bg

        # threshold foreground    
        thresh_val = threshold_otsu(img_fg)

        op = copy.deepcopy(img_fg)

        op[img_fg < thresh_val] = 0.
        op[img_fg >= thresh_val] = 255.

        return op