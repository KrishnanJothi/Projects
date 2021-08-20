#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os.path
from scipy.spatial import Delaunay
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from utils import *


# Load images.
src_img, dst_img = [convertColorImagesBGR2RGB(cv2.imread(p)) for p in ("img/face1.jpg", "img/face2.jpg")]

if src_img.shape[1::-1] != dst_img.shape[1::-1]:
    print("Resolution of images does not match!")
    exit()

# Rescale images.
max_img_size = 512
size = src_img.shape[1::-1]
if max(size) > max_img_size:
    size = np.dot(size, max_img_size / max(size)).astype(np.int).tolist()
    src_img, dst_img = [cv2.resize(img[...,:3], size) if img.shape[1::-1] != size else img for img in (src_img, dst_img)]
src_img = src_img[:,:-50]
dst_img = dst_img[:,:-50]
w, h = src_img.shape[1::-1]


# Find 68 landmark dlib face model.
predictor_file = "shape_predictor_68_face_landmarks.dat"
predictor_path = "models/" + predictor_file
if not os.path.isfile(predictor_path):
    print("File not found: %s\nDownload from http://dlib.net/files/%s.bz2"%(predictor_path, predictor_file))
    exit()



#
# Task 1
#
# Complete the code for face morphing.

def weighted_average(img1, img2, alpha = .5):
    # TODO: Compute and return the weighted average (linear interpolation) of the two supplied images.
    #  Use the interpolation factor `alpha` such that the function returns `img1` if `alpha` == 0, `img2` if `alpha` == 1, and the interpolation otherwise.
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    return dst


def get_face_landmarks(image, predictor_path = predictor_path):
    # TODO: Use the `dlib` library for "Face Landmark Detection".
    #  The function shall return a numpy array of shape (68, 2), holding 68 larndmarks as 2D integer pixel position.
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(predictor_path)
    faces = face_detector(image, 1)

    landmark_tuple = []
    for k, d in enumerate(faces):
        landmarks = landmark_detector(image, d)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))

    return np.array(landmark_tuple)


def weighted_average_points(src_points, dst_points, alpha = .5):
    # TODO: Compute and return the weighted average (linear interpolation) of the two sets of supplied points.
    #  Use the interpolation factor `alpha` such that the function returns `start_points` if `alpha` == 0, `end_points` if `alpha` == 1, and the interpolation otherwise.
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(src_points, alpha, dst_points, beta, 0.0)
    return dst


# Warp each triangle from the `src_img` to the destination image.
def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    # Generate x,y pixel coordinates
    pixel_coords = np.asarray([(x, y) for y in range(src_img.shape[0]-2) for x in range(src_img.shape[1]-2)], np.uint32)
    # Indices to vertices. -1 if pixel is not in any triangle.
    triangle_indices = delaunay.find_simplex(pixel_coords)

    # # DEBUG visualization of triangle surfaces.
    # triangle_surfaces = np.reshape(triangle_indices, (pixel_coords[-1][1] - pixel_coords[0][1] + 1, pixel_coords[-1][0] - pixel_coords[0][0] + 1))
    # showImage(triangle_surfaces.astype(np.uint8))

    for simplex_index in range(len(delaunay.simplices)):
        coords = pixel_coords[triangle_indices == simplex_index]
        num_coords = len(coords)
        if num_coords > 0:
            out_coords = np.dot(tri_affines[simplex_index], np.vstack((coords.T, np.ones(num_coords))))
            x, y = coords.T
            result_img[y, x] = bilinear_interpolate(src_img, out_coords)

# Calculate the affine transformation matrix for each triangle vertex (x,y) from `dest_points` to `src_points`.
def gen_triangular_affine_matrices(vertices, src_points, dest_points):
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat

def warp_image(src_img, src_points, dest_points):
    result_img = src_img.copy()
    delaunay = Delaunay(dest_points)
    tri_affines = np.asarray(list(gen_triangular_affine_matrices(delaunay.simplices, src_points, dest_points)))
    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)
    return result_img, delaunay


# Detect facial landmarks as control points for warps.
src_points, dst_points = [get_face_landmarks(img) for img in (src_img, dst_img)]


#
# Task 2
#
# Improve morphing output.

# TODO: Extend the `src_points` and `dst_points` arrays such that also the surrounding parts of the images are warped.
#  Hint: The corners of both images shall note move when warping.
src_points, dst_points = [np.concatenate((points, [(0, 0), (w-2, 0), (0, h-2), (w-2, h-2)])) for points in (src_points, dst_points)]

def bilinear_interpolate(img, coords):
    # TODO: Implement bilinear interpolation.
    #  The function shall return an array of RGB values that correspond to the interpolated pixel colors in `img` at the positions in `coords`.
    #  As the coords are floating point values, use bilinear interpolation, such that the RGB color for each position in `coords` is interpolated from 4 instead of just one pixels in `img`.
    int_coords = coords.astype(np.int)
    x0, y0 = int_coords
    dx, dy = coords - int_coords
    # Get 4 nearest pixels
    q11, q21, q12, q22 = img[y0, x0], img[y0, x0+1], img[y0+1, x0], img[y0+1, x0+1]
    # Interpolate in X
    top = q11.T * (1 - dx) + q21.T * dx
    bottom = q12.T * (1 - dx) + q22.T * dx
    # Interpolate in Y
    interpolated_pixels = top * (1 - dy) + bottom * dy
    return interpolated_pixels.T


found_face_points = len(src_points) > 0 and len(dst_points) > 0

fig = plt.figure(figsize=(16,8))
_, imgAxs1 = showImages([("img", src_img), ("Facial landmarks", src_img), ("Delaunay triangulation", src_img),
                                 dst_img,                       dst_img,                             dst_img], 3, show_window_now=False, convertRGB2BGR=False)
imgAxs1[0].text(-30, h/2, "src", rotation="vertical", va="center")
imgAxs1[3].text(-30, h/2, "dst", rotation="vertical", va="center")

if found_face_points:
    imgAxs1[1].plot(src_points[:,0], src_points[:,1], 'o', markersize=3)
    imgAxs1[4].plot(dst_points[:,0], dst_points[:,1], 'o', markersize=3)

    alpha=.5
    points = weighted_average_points(src_points, dst_points, alpha)
    _, src_delaunay = warp_image(src_img, src_points, points)
    _, dst_delaunay = warp_image(dst_img, dst_points, points)

    imgAxs1[2].triplot(src_points[:,0], src_points[:,1], src_delaunay.simplices.copy())
    imgAxs1[5].triplot(dst_points[:,0], dst_points[:,1], dst_delaunay.simplices.copy())


imgs = []
num_imgs = 16
if found_face_points:
    for alpha in np.linspace(0, 1, num_imgs):
        print("progress: %.2f"%alpha)
        points = weighted_average_points(src_points, dst_points, alpha)
        src_face, _ = warp_image(src_img, src_points, points)
        dst_face, _ = warp_image(dst_img, dst_points, points)
        imgs.append((src_face, weighted_average(src_face, dst_face, alpha), dst_face))

imgs_to_show = [("src", src_img), ("avg", src_img), ("dst", dst_img)]
if found_face_points:
    imgs_to_show += [imgs[0][0], imgs[0][1], imgs[0][2]]

fig = plt.figure(figsize=(16,8))
imgRefs2, imgAxs2 = showImages(imgs_to_show, 3, show_window_now=False, convertRGB2BGR=False, padding=(0, .1, 0, .05))

imgAxs2[0].text(-30, h/2, "blend", rotation="vertical", va="center")
if found_face_points:
    imgAxs2[3].text(-30, h/2, "warp + blend", rotation="vertical", va="center")

def updateImgs(percent):
    alpha = percent/100
    imgRefs2[1].set_data(weighted_average(src_img, dst_img, alpha))
    if found_face_points:
        selectedImgs = imgs[int(round((num_imgs-1) * alpha))]
        imgRefs2[3].set_data(selectedImgs[0])
        imgRefs2[4].set_data(selectedImgs[1])
        imgRefs2[5].set_data(selectedImgs[2])

ax_slider = plt.axes([.33, .01, .33, .05])
slider = Slider(ax=ax_slider, label='Percent', valmin=0, valmax=100, valinit=50, valstep=100/num_imgs)
slider.on_changed(updateImgs)
updateImgs(50)

plt.show()
