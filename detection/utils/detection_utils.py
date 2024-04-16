from PIL import Image
import numpy as np
import os


def cut_xy(orig_x, orig_y, orig_w, orig_h, cut_diff):
    '''
    Shrink the image proportionally, in yolo coords
    '''
    orig_x += cut_diff
    orig_y += cut_diff
    orig_w -= cut_diff * 2
    orig_h -= cut_diff * 2
    return orig_x, orig_y, orig_w, orig_h
