import os
import cv2
import numpy as np
from pandas import concat
from imageio import loads_pil, save_cv2

SQUARES = {4:2, 9:3, 16:4, 25:5}

def resize_hw(im, height, width, interpolation_method=cv2.INTER_AREA):
    dim = (width, height)
    resized = cv2.resize(im, dim, interpolation=interpolation_method)
    return resized

def concat_square(images, dim, method=cv2.INTER_AREA):
    _map = []
    n = len(images)
    assert n in SQUARES.keys() #require that n is a perfect square
    root = SQUARES[n] #root square root of n
    hw = int(dim / root) #dim = 512 for us?
    rows = []
    c = 0
    k = 0
    row = []
    for i in range(n):
        if c == root:
            cat_row = cv2.hconcat(row)
            rows.append(cat_row)
            row = []
            c = 0
            k += 1
        row.append(resize_hw(images[i], hw, hw, method))
        _map.append({'id': i, 'origin': (k*hw, c*hw), 'w': hw, 'h': hw}) # origin is (r, c) location
        c += 1
    cat_row = cv2.hconcat(row)
    rows.append(cat_row)
    output = cv2.vconcat(rows)
    print(_map)
    return output

# im1 = cv2.imread('data/sample_barcodes/001.jpg')
# im_v = cv2.vconcat([im1, im1, im1])
# im_tile = np.tile(im1, (2, 2, 1))
# im_tile

# cv2.imwrite('test.jpg', im_tile)

# fnames = [os.path.join('data/sample_barcodes', fname) for fname in os.listdir('data/sample_barcodes')]
# ims = [np.array(im) for im in loads_pil(fnames)]
# im = concat_square(ims, 512)
# save_cv2(im, '_cat.jpg')