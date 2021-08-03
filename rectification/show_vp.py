from math import pi
import numpy as np
from numpy.core.fromnumeric import resize

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
from matplotlib import image
from matplotlib import pyplot as plt

def proj2cart(vp):
    zero = np.finfo(np.float64).tiny
    if vp[2] == 0:
        vp = np.array([vp[0]/zero, vp[1]/zero])
        return vp
    else:
        vp = np.array([vp[0]/vp[2], vp[1]/vp[2]])
        return vp



working_image = imread("img/img20.jpg")

height = working_image.shape[0]
width = working_image.shape[1]

vanish_points = np.load('npy/vanish_points.npy', allow_pickle = True)

vp1 = vanish_points[0]/np.linalg.norm(vanish_points[0])
vp2 = vanish_points[1]/np.linalg.norm(vanish_points[1])

print(vp1,vp2)

cartesian_vp1 = proj2cart(vp1)
cartesian_vp2 = proj2cart(vp2)

print(cartesian_vp1,cartesian_vp2)

vertical_debug = np.load('npy/vertical_debug_lines.npy', allow_pickle = True)
horizontal_debug = np.load('npy/horizontal_debug_lines.npy', allow_pickle = True)

canvas = image.imread('img/img20.jpg')

for l in vertical_debug:
    angle = l[1]
    dist = l[2]
    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='r')


for l in horizontal_debug:
    angle = l[1]
    dist = l[2]
    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='b')

'''m1 = (cartesian_vp1[1] - height/2)/(cartesian_vp1[0] - width/2)
m2 = (cartesian_vp2[1] - height/2)/(cartesian_vp2[0] - width/2)

plt.axline((width/2, height/2), slope=m1, linewidth=1, linestyle='--', dashes=(5, 3), color='y')
plt.axline((width/2, height/2), slope=m2, linewidth=1, linestyle='--', dashes=(5, 3) , color='y')'''

plt.axline((width/2, height/2), (cartesian_vp1[0], cartesian_vp1[1]) , linewidth=1, linestyle='--', dashes=(5, 3), color='y')
plt.axline((width/2, height/2), (cartesian_vp2[0], cartesian_vp2[1]), linewidth=1, linestyle='--', dashes=(5, 3) , color='g')



plt.imshow(canvas)
plt.savefig("img/vanish_points.jpg", dpi = 1500)
plt.clf()



