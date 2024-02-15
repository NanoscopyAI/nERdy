import imageio
import skimage
from skimage.filters import threshold_local
from skimage import exposure
import argparse
import matlab.engine as m_engine

# preprocessing step in nERdy
def preprocess(img_path):
    """
    Preprocesses the image by applying a CLAHE filter and morphological operations
    """
    img = imageio.imread(img_path)
    std_img = (img - img.min()) / (img.max() - img.min())
    histeq_img = exposure.equalize_adapthist(std_img)
    aop_img = skimage.morphology.area_opening(histeq_img, area_threshold=2)
    erod_img = skimage.morphology.erosion(aop_img)
    loc = threshold_local(erod_img, 3)
    return loc

def runner():
    """
    Preprocessing followed by vessel enhancement
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="Name of the file to be processed")

    args = parser.parse_args()

    filename = args.filename

    loc = preprocess(filename)

    imageio.imsave("preprocessed_" + filename, loc)

    Engine = m_engine.start_matlab()

    Engine.Vessel2d("preprocessed_" + filename)

    Engine.quit()

