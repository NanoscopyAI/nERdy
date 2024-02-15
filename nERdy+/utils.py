import imageio
from skimage import morphology

# create mask from skeleton
def mask_creator():
    skel = imageio.imread('climp12_proc_skel.png')
    mask = morphology.dilation(skel)

    # imageio.imwrite('climp12_proc_skel_mask.png', mask)
    return mask

# create adaptive mask 
def adapative_mask_creator():
    skel = imageio.imread('climp12_proc_skel.png')
    inv = ~skel
    dilskel = morphology.dilation(skel)
    closed_skel = morphology.area_closing(skel)
    roi = closed_skel - inv

    roi[roi==255] = 0
    roi[roi==254] = 255

    op = dilskel - roi

    # imageio.imwrite('climp12_proc_skel_mask.png', op)
    return op