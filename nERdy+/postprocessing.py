from skimage import restoration
from skimage.filters import threshold_otsu
import copy

def postprocessing(img):
    """
    Perform post-processing on the input image.

    Parameters:
    img (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Processed image.
    """
    # Apply rolling ball algorithm to get background
    img_bg = restoration.rolling_ball(img)

    # Subtract background from input image to get foreground
    img_fg = img - img_bg

    # Threshold the foreground image
    thresh_val = threshold_otsu(img_fg)

    # Create a copy of the foreground image
    op = copy.deepcopy(img_fg)

    # Set pixels below the threshold to 0 and above the threshold to 255
    op[img_fg < thresh_val] = 0.
    op[img_fg >= thresh_val] = 255.

    return op