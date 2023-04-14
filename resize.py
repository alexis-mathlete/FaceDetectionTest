import cv2
import numpy as np
from PIL import Image

from imageio import load_cv2, cv2_to_pil, save_cv2
from colors import COLORS

cv2_interpolations = {
    "inter_cubic": cv2.INTER_CUBIC,
    "inter_nearest": cv2.INTER_NEAREST,
    "inter_linear": cv2.INTER_LINEAR,
    "inter_lanczos4": cv2.INTER_LANCZOS4,
    "inter_area": cv2.INTER_AREA,
}
DEFAULT_INTER = "inter_linear"
DEFAULT_BORDER_WIDTH = 50
DEFAULT_BORDER_COLOR = "blue"

def resize_hw(im, height, width, interpolation_method=DEFAULT_INTER):
    dim = (width, height)
    inter_method = cv2_interpolations[interpolation_method]
    resized = cv2.resize(im, dim, interpolation=inter_method)
    return resized

def resize_maxheight(im, maxheight, interpolation_method=DEFAULT_INTER):
    (h,w,c) = im.shape
    ratio = int(maxheight / h)
    width = ratio * w
    return resize_hw(im, maxheight, width, interpolation_method)

def resize_maxwidth(im, maxwidth, interpolation_method=DEFAULT_INTER):
    (h,w,c) = im.shape
    ratio = int(maxwidth / w)
    height = ratio * h
    return resize_hw(im, height, maxwidth, interpolation_method)

def add_border(
        im,
        border_width=DEFAULT_BORDER_WIDTH,
        border_color=DEFAULT_BORDER_COLOR,
    ):
    color = COLORS[border_color]
    top, bottom, left, right = [border_width] * 4
    im_with_border = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return im_with_border

def pad_sides(im, pad_width, pad_color="white"):
    (h,w,c) = im.shape
    canvas = blank_canvas(h, w+(2*pad_width), c, pad_color)
    canvas[:, pad_width:pad_width+w] = im
    return canvas

def pad_topbot(im, pad_height, pad_color="white"):
    (h,w,c) = im.shape
    canvas = blank_canvas(h+(2*pad_height), w, c, pad_color)
    canvas[pad_height:pad_height+h, :] = im
    return canvas

def blank_canvas(h, w, c, color="white"):
    if color == "white":
        _color = 255
    elif color == "black":
        _color = 0
    canvas = np.ones((h,w,c)) * _color
    return canvas

def shrink_and_pad(
        im,
        to_height,
        to_width,
        frame_height,
        frame_width,
        x,
        y,
        canvas_color="black",
        interpolation_method=DEFAULT_INTER,
    ):
    """
    to_height & to_width: size for resizing im
    frame_height & frame_width: size of frame in which resized im will be pasted
    x & y: x and y coordinates where resized im upper left corner will be pasted
        - note: the origin (0,0) is in the upper left
    """
    res_im = resize_hw(im, to_height, to_width, interpolation_method)
    (h,w,c) = res_im.shape
    canvas = blank_canvas(frame_height, frame_width, c, canvas_color)
    canvas[y:y+h, x:x+h] = res_im
    return canvas

def pad_to_square(cv2_im):
    old_size = cv2_im.shape[:2]
    desired_size = max(old_size)
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(cv2_im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im


# for file_path in file_path_l:
#     pad_to_square(file_path)
#print(im)
#print(im.shape)
#cv2.imshow('image', im)
#cv2.waitKey(0)
#im = add_border(im)
#im = resize_maxwidth(im, 1000)
# im = pad_sides(im, 240, "black")
# #im = shrink_and_pad(im, 200, 200, 600, 600, 50, 300)
# #cv2_to_pil(im).show()
# cv2.imshow('image', im)
# cv2.waitKey(0)
# save_cv2(im, 'tmp.jpg')