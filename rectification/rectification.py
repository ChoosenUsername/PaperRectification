from os import PRIO_PGRP
from re import X
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import clip, resize
from numpy.lib import angle
from skimage import transform
from skimage import color

from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage.draw import circle,disk
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


canvas = image.imread('img/img1.jpg')

start_image = imread("img/img1.jpg")
res = start_image


h = start_image.shape[0]
w = start_image.shape[1]

vertical = np.load('npy/vertical_debug_lines.npy', allow_pickle = True)
horizontal = np.load('npy/horizontal_debug_lines.npy', allow_pickle=True)

vanish_points = np.load('npy/vanish_points.npy', allow_pickle=True)

horizontal_temp_line_index = np.random.choice(horizontal.shape[0], 1)
vertical_temp_line_index = np.random.choice(vertical.shape[0], 1)

###################################################################################

dots_on_horizontal_line = []
horizontal_temp_line = horizontal[horizontal_temp_line_index,:]
angle = horizontal_temp_line[0,1]
dist = horizontal_temp_line[0,2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='r')
b1 = y0 - x0*np.tan(angle + np.pi/2)
a1 = np.tan(angle + np.pi/2)

for line in vertical:

    angle = line[1]
    dist = line[2]
    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='r')
    b2 = y0 - x0*np.tan(angle + np.pi/2)
    a2 = np.tan(angle + np.pi/2)


    x1 = (b2 - b1)/(a1-a2)
    y1 = a1*(b2-b1)/(a1-a2) + b1

    dots_on_horizontal_line.append([x1,y1])



dots_on_horizontal_line = np.array(dots_on_horizontal_line)

most_left_dot_index = np.argmin(dots_on_horizontal_line[:,0])
most_left_dot = dots_on_horizontal_line[most_left_dot_index]


most_right_dot_index = np.argmax(dots_on_horizontal_line[:,0])
most_right_dot = dots_on_horizontal_line[most_right_dot_index]


###################################################################################

dots_on_vertical_line = []
vertical_temp_line = vertical[vertical_temp_line_index,:]
angle = vertical_temp_line[0,1]
dist = vertical_temp_line[0,2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='r')
b1 = y0 - x0*np.tan(angle + np.pi/2)
a1 = np.tan(angle + np.pi/2)

for line in horizontal:

    angle = line[1]
    dist = line[2]
    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='r')
    b2 = y0 - x0*np.tan(angle + np.pi/2)
    a2 = np.tan(angle + np.pi/2)


    x1 = (b2 - b1)/(a1-a2)
    y1 = a1*(b2-b1)/(a1-a2) + b1

    dots_on_vertical_line.append([x1,y1])


dots_on_vertical_line = np.array(dots_on_vertical_line)

most_top_dot_index = np.argmin(dots_on_vertical_line[:,1])
most_top_dot = dots_on_vertical_line[most_top_dot_index]

most_bot_dot_index = np.argmax(dots_on_vertical_line[:,0])
most_bot_dot = dots_on_vertical_line[most_bot_dot_index]

###################################################################################


left_line = vertical[most_left_dot_index]
angle = left_line[1]
dist = left_line[2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='b')
b1 = y0 - x0*np.tan(angle + np.pi/2)
a1 = np.tan(angle + np.pi/2)

right_line = vertical[most_right_dot_index]
angle = right_line[1]
dist = right_line[2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='b')
b2 = y0 - x0*np.tan(angle + np.pi/2)
a2 = np.tan(angle + np.pi/2)

low_line = horizontal[most_bot_dot_index]
angle = low_line[1]
dist = low_line[2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='b')
b3 = y0 - x0*np.tan(angle + np.pi/2)
a3 = np.tan(angle + np.pi/2)

upper_line = horizontal[most_top_dot_index]
angle = upper_line[1]
dist = upper_line[2]
a,b = np.cos(angle), np.sin(angle)
x0, y0 = dist * np.array([a, b])
plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='b')
b4 = y0 - x0*np.tan(angle + np.pi/2)
a4 = np.tan(angle + np.pi/2)


#left top
x1 = (b4 - b1)/(a1-a4)
y1 = a1*(b4-b1)/(a1-a4) + b1

#right top
x2 = (b2 - b4)/(a4-a2)
y2 = a4*(b2-b4)/(a4-a2) + b4

#left bot
x3 = (b1 - b3)/(a3-a1)
y3 = a3*(b1-b3)/(a3-a1) + b3

#right bot
x4 = (b2 - b3)/(a3-a2)
y4 = a3*(b2-b3)/(a3-a2) + b3

plt.imshow(canvas)
plt.savefig("img/border.jpg", dpi = 1500)
plt.clf()



src = np.array([ [x3,y3], [x1,y1], [x4,y4], [x2,y2] ])
dst = np.array([ [0,h-1], [0,0], [w-1,h-1], [w-1,0] ])

tform = transform.estimate_transform('projective', src, dst)
tf_img = transform.warp(start_image, tform.inverse)

imsave("img/result.jpg", img_as_ubyte(tf_img))

