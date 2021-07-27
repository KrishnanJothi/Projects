
import cv2
import matplotlib.pyplot as plt
import numpy as np

def showImage(img, show_window_now = True, convertRGB2BGR = True, _ax = None):
    img = convertColorImagesBGR2RGB(img) if convertRGB2BGR else img
    is_color_img = isColorImage(img)

    if _ax is None:
        plt_img = plt.imshow(img, interpolation='antialiased', cmap=None if is_color_img else 'gray')
        plt.axis('off')
        plt.tight_layout()
    else:
        plt_img = _ax.imshow(img, interpolation='antialiased', cmap=None if is_color_img else 'gray')
        _ax.axis('off')
    if show_window_now:
        plt.show()
    return plt_img

def showImages(imgs, num_cols = None, show_window_now = True, convertRGB2BGR = True, transpose = False, spacing = None, padding = None):
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
    axs = []
    i = 0
    tmp_imgs = []
    for img in imgs:
        tmp = type('', (), {})()
        if num_cols is None:
            tmp.pos = (0, i)
        else:
            tmp.pos = (i // num_cols, i % num_cols)
        if transpose:
            tmp.pos = tmp.pos[::-1]
        tmp.img = img
        tmp.title = None
        tmp.span = (1, 1)
        if img is not None:
            if type(img) is tuple:
                tmp.img = img[1]
                tmp.title = img[0]
                if len(img) > 2:
                    tmp.span = img[2]

            i += tmp.span[0] * tmp.span[1]
        else:
            i += 1
        tmp_imgs.append(tmp)


    if num_cols is None:
        grid = (1, i)
    else:
        num_rows = (i - 1) // num_cols + 1
        grid = (num_rows, num_cols)

    if transpose:
        grid = grid[::-1]

    for img in tmp_imgs:
        if img.img is not None or img.title is not None:
            ax = plt.subplot2grid(grid, img.pos, colspan=img.span[0], rowspan=img.span[1])
            axs.append(ax)
            if img.img is not None:
                plt_imgs.append(showImage(img.img, False, convertRGB2BGR=convertRGB2BGR, _ax=ax))
            if img.title is not None:
                ax.set_title(img.title)

    plt.tight_layout()
    if spacing is not None:
        plt.subplots_adjust(wspace=spacing[0], hspace=spacing[1])
    if padding is not None:
        plt.subplots_adjust(left=padding[0], bottom=padding[1], right=1 - padding[2], top=1 - padding[3])
    if show_window_now:
        plt.show()
    return plt_imgs, axs


def isColorImage(img):
    return len(img.shape) == 3 and img.shape[2] == 3

def convertColorImagesBGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

def PLACEHOLDER_IMG(img):
    return img.copy()

def REPLACE_THIS(input):
    return input

def REPLACE_THIS_MODEL(input):
    return 1
