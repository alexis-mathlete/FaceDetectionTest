import sys
from PIL import Image, ImageFilter
from torchvision import transforms
from rgbtransforms import RGBTransform

def blur(im):
    im = im.convert('RGB')
    im_bl = im.filter(ImageFilter.BLUR)
    return im_bl

def grayscale(im, alpha=False):
    if alpha:
        im_gr = im.convert('LA')
    else:
        im_gr = im.convert('L')
    return im_gr

def solarize(im, value=100):
    im = im.convert('RGB')
    im_sl = transforms.functional.solarize(im, value)
    return im_sl

def equalize(im):
    im = im.convert('RGB')
    im_eq = transforms.functional.equalize(im)
    return im_eq

def invert(im):
    im = im.convert('RGB')
    im_iv = transforms.functional.invert(im)
    return im_iv

def tint(im, color='blue'):
    im = im.convert('RGB')
    if color == 'red':
        im_tn = RGBTransform().mix_with((255,0,0), factor=0.3).applied_to(im)
    elif color == 'green':
        im_tn = RGBTransform().mix_with((0,255,0), factor=0.3).applied_to(im)
    elif color == 'blue':
        im_tn = RGBTransform().mix_with((0,0,255), factor=0.3).applied_to(im)
    elif color == 'yellow':
        im_tn = RGBTransform().mix_with((255,255,0), factor=0.3).applied_to(im)
    elif color == 'teal':
        im_tn = RGBTransform().mix_with((0,255,255), factor=0.3).applied_to(im)
    elif color == 'pink':
        im_tn = RGBTransform().mix_with((255,0,255), factor=0.3).applied_to(im)
    else:
        print('tint color "{}" not supported'.format(color))
    return im_tn

if __name__ == "__main__":
    MODIFICATIONS = {
        'blur': blur,
        'grayscale': grayscale,
        'solarize': solarize,
        'equalize': equalize,
        'invert': invert,
        'tint': tint,
    }
    imfile = sys.argv[1]
    im = Image.open(imfile)
    mod = sys.argv[2]
    if mod in MODIFICATIONS:
        im_mod = MODIFICATIONS[mod](im)
        im_mod.show()