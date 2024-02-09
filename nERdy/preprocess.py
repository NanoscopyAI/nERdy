import imageio
import skimage
from skimage.filters import threshold_local
from skimage import exposure


def preprocess(img):
    """
    Preprocesses the image by applying a CLAHE filter and a morphological operations
    """
    img = (img - img.min()) / (img.max() - img.min())
    img = exposure.equalize_adapthist(img)
    aop = skimage.morphology.area_opening(img, area_threshold=2)
    erod = skimage.morphology.erosion(aop)
    loc = threshold_local(erod, 3)
    return loc

