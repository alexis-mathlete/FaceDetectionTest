import cv2
import numpy as np
from PIL import Image
from skimage import io as skio

def load_pil(imfile):
    im = Image.open(imfile)
    return im

def loads_pil(imfiles):
    ims = []
    for imfile in imfiles:
        ims.append(load_pil(imfile))
    return ims

def save_pil(im, imfile):
    im.save(imfile)

def load_cv2(imfile):
    im = cv2.imread(imfile)
    return im

def save_cv2(im, imfile):
    cv2.imwrite(imfile, im)

def load_skimage(imfile):
    im = skio.imread(imfile)

def save_skimage(im, imfile):
    skio.imsave(imfile, im)

def cv2_to_pil(im):
    _im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return Image.fromarray(_im)

def pil_to_cv2(im):
    return np.asarray(im)