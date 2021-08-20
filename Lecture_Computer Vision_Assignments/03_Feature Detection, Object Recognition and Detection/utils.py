
import cv2
import matplotlib.pyplot as plt
import numpy as np

def showImage(img, show_window_now = True):
    img, color_img = convertColorImagesBGR2RGB(img)
    plt_img = plt.imshow(img, interpolation='antialiased', cmap=None if color_img else 'gray')
    plt.axis('off')
    plt.tight_layout()
    if show_window_now:
        plt.show()
    return plt_img

def showImages(imgs, num_cols = None, show_window_now = True, transpose = False, spacing = (.05,)*2, padding = (.01,)*4):
    '''
    imgs:
        [image|('caption', image)|None, ...]
        list of images
    num_cols:
        int | None
    transpose:
        True | False
        flip rows and columns
    show_window_now:
        True | False
    spacing:
        (int, int)
        horizontal and vertical spacing between images
    padding:
        (int, int, int, int)
        left, bottom, right, top paddding
    '''
    plt_imgs = []
    i = 0
    for img in imgs:
        i = i + 1
        if (img is not None):
            if num_cols is None:
                if transpose:
                    plt.subplot(len(imgs), 1, i)
                else:
                    plt.subplot(1, len(imgs), i)
            else:
                num_rows = (len(imgs) - 1) // num_cols + 1
                row = (i - 1) // num_cols
                col = (i - 1) % num_cols
                if transpose:
                    plt.subplot2grid((num_cols, num_rows), (col, row))
                else:
                    plt.subplot2grid((num_rows, num_cols), (row, col))
            if type(img) is tuple:
                plt.gca().set_title(img[0])
                plt_imgs.append(showImage(img[1], False))
            else:
                plt_imgs.append(showImage(img, False))
    plt.subplots_adjust(left=padding[0], bottom=padding[1], right=1 - padding[2], top=1 - padding[3], wspace=spacing[0], hspace=spacing[1])
    plt.tight_layout()
    if show_window_now:
        plt.show()
    return plt_imgs

def convertColorImagesBGR2RGB(img):
    is_color_img = len(img.shape) == 3 and img.shape[2] == 3
    if is_color_img:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, True
    return img, False

def from0_1to0_255asUint8(float_img):
    img = float_img * 255
    return asUint8(img)

def clip0_255asUint8(float_img):
    img = float_img.copy()
    np.clip(float_img, 0, 255, img)
    return asUint8(img)

def asUint8(float_img):
    return float_img.astype(np.uint8)

def PLACEHOLDER(img):
    return np.zeros(img.shape, np.uint8)

REPLACE_THIS = 1

def REPLACE_THIS_MODEL(input):
    return 1
