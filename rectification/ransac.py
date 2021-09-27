from os import PRIO_PGRP
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import clip, resize
from numpy.lib import angle
from numpy.lib.function_base import diff
from numpy.linalg import norm
from skimage import transform
from skimage import filters

from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage.draw import circle
from skimage import data
from skimage.filters import gaussian,threshold_niblack
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.morphology import skeletonize
from skimage.io import imsave
import cv2 as cv
from matplotlib import image

def h2c(vp):
    zero = np.finfo(np.float64).tiny
    if vp[2] == 0:
        vp = np.array([vp[0]/zero, vp[1]/zero])
        return vp
    else:
        vp = np.array([vp[0]/vp[2], vp[1]/vp[2]])
        return vp

def h2c_for_set(vp):
    zero = np.finfo(np.float64).tiny
    ind = np.where(vp[:,2] == 0)
    vp[ind, 2] = zero
    vp = vp/vp[2]
    return vp[:,:2]

#############################
# Vertical lines filtration #
#############################

start_image = imread("/home/pavel/PaperRectification/img/img36.jpg")

output_image = start_image

height = start_image.shape[0]
width = start_image.shape[1]

n = start_image.shape[0]
m = start_image.shape[1]

cross_vertical = np.load('/home/pavel/PaperRectification/npy/cross_vertical.npy', allow_pickle=True)


current_amount = 0
best_mask = 0
vanish_point = None

first_lines_set = cross_vertical[ np.random.choice(cross_vertical.shape[0], 100000) ]
second_lines_set = cross_vertical[ np.random.choice(cross_vertical.shape[0], 100000) ]

intersection_table = np.concatenate((first_lines_set, second_lines_set, list(np.cross( np.stack( first_lines_set[:,0], axis=0 ),  np.stack( second_lines_set[:,0], axis=0)))), axis=1)

#удаление векторов вида (0,0,0)
indexes_of_not_valid_vectors = np.where(  np.linalg.norm(np.float64(intersection_table[:,6:9]), axis=1) == 0 )
intersection_table = np.delete(intersection_table, indexes_of_not_valid_vectors, axis=0)

check_points = intersection_table[: , 6 : 9] # n возможных точек схода

# для каждой линии для которой есть точка схода, ранее отфильтрованная, создается набор точкек,
# который строится векторным произведением всех найденных линий на изображении и линией из предполагаемой точки схода
filter = lambda point: point[ np.where(np.linalg.norm(point,axis=1) != 0 ) ]
index = lambda point: np.where(np.linalg.norm(point,axis=1) != 0 )
satelite_points = np.array([ (filter(np.cross(np.stack(cross_vertical[:,0], axis=0), i))) for i in intersection_table[:,0] ])
upd_vertical = np.array( [  cross_vertical[:,:][index(np.cross(np.stack(cross_vertical[:,0], axis=0), i))]  for i in intersection_table[:,0]] )


epsilon = 3000
for i,j,k in zip(satelite_points, check_points, upd_vertical):

    c1 = h2c_for_set(i)
    c2 = h2c(j)
    
    dist = np.linalg.norm(c1 - c2, axis = 1)
    print(np.average(dist))

    mask = dist < epsilon

    amount = mask.sum()
    if amount > current_amount:
        current_amount = amount
        vanish_point = j
        best_mask = mask
        vertical = k
        closes_line_index = np.argmin(dist)


debug_lines = vertical[best_mask]
np.save( '/home/pavel/PaperRectification/npy/vertical_debug_lines', debug_lines)


vp1 = vanish_point
print(current_amount, 'for', vanish_point)


canvas = image.imread('/home/pavel/PaperRectification/img/img36.jpg')

for l in debug_lines:
    angle = l[1]
    dist = l[2]
    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='r')

closest_line = debug_lines[closes_line_index]
angle = closest_line[1]
dist = closest_line[2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='y')


vp1 = vp1/np.linalg.norm(vp1)
cartesian_vp1 = h2c(vp1)
plt.axline((width/2, height/2), (cartesian_vp1[0], cartesian_vp1[1]) , linewidth=1, linestyle='--', dashes=(5, 3), color='g')

plt.imshow(canvas)
plt.savefig("/home/pavel/PaperRectification/img/vertical_debug_lines.jpg", dpi = 1500)
plt.clf()

###############################
# Horizontal lines filtration #
###############################

start_image = imread("/home/pavel/PaperRectification/img/img36.jpg")

output_image = start_image

height = start_image.shape[0]
width = start_image.shape[1]

n = start_image.shape[0]
m = start_image.shape[1]

cross_horizontal = np.load('/home/pavel/PaperRectification/npy/cross_horizontal.npy',allow_pickle=True)


current_amount = 0
best_mask = 0
vanish_point = None

first_lines_set = cross_horizontal[ np.random.choice(cross_horizontal.shape[0], 100000) ]
second_lines_set = cross_horizontal[ np.random.choice(cross_horizontal.shape[0], 100000) ]

intersection_table = np.concatenate((first_lines_set, second_lines_set, list(np.cross( np.stack( first_lines_set[:,0], axis=0 ),  np.stack( second_lines_set[:,0], axis=0)))), axis=1)

#удаление векторов вида (0,0,0)
indexes_of_not_valid_vectors = np.where(  np.linalg.norm(np.float64(intersection_table[:,6:9]), axis=1) == 0 )
intersection_table = np.delete(intersection_table, indexes_of_not_valid_vectors, axis=0)

check_points = intersection_table[: , 6 : 9] # n возможных точек схода

# для каждой линии для которой есть точка схода, ранее отфильтрованная, создается набор точкек,
# который строится векторным произведением всех найденных линий на изображении и линией из предполагаемой точки схода
filter = lambda point: point[ np.where(np.linalg.norm(point,axis=1) != 0 ) ]
index = lambda point: np.where(np.linalg.norm(point,axis=1) != 0 )
satelite_points = np.array([ (filter(np.cross(np.stack(cross_horizontal[:,0], axis=0), i))) for i in intersection_table[:,0] ])
upd_horizontal = np.array( [  cross_horizontal[:,:][index(np.cross(np.stack(cross_horizontal[:,0], axis=0), i))]  for i in intersection_table[:,0]] )


epsilon = 6000
for i,j,k in zip(satelite_points, check_points, upd_horizontal):

    c1 = h2c_for_set(i)
    c2 = h2c(j)
    
    dist = np.linalg.norm(c1 - c2, axis = 1)
    print(np.average(dist))

    mask = dist < epsilon

    amount = mask.sum()
    if amount > current_amount:
        current_amount = amount
        vanish_point = j
        best_mask = mask
        horizontal = k
        closes_line_index = np.argmin(dist)

debug_lines = horizontal[best_mask]
np.save( '/home/pavel/PaperRectification/npy/horizontal_debug_lines', debug_lines)


vp2 = vanish_point
print(current_amount, 'for', vanish_point)


canvas = image.imread('/home/pavel/PaperRectification/img/img36.jpg')
for l in debug_lines:
    angle = l[1]
    dist = l[2]
    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='r')


closest_line = debug_lines[closes_line_index]
angle = closest_line[1]
dist = closest_line[2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='y')


vp2 = vp2/np.linalg.norm(vp2)
cartesian_vp2 = h2c(vp2)
plt.axline((width/2, height/2), (cartesian_vp2[0], cartesian_vp2[1]) , linewidth=1, linestyle='--', dashes=(5, 3), color='g')


plt.imshow(canvas)
plt.savefig("/home/pavel/PaperRectification/img/horizontal_debug_lines.jpg", dpi = 1500)


np.save( '/home/pavel/PaperRectification/npy/vanish_points', np.array([vp1, vp2]))

