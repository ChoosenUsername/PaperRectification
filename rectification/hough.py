import numpy as np
from numpy.core.fromnumeric import resize

from skimage.io import imread
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage.draw import circle
from skimage import data
from skimage.filters import gaussian,threshold_niblack,threshold_sauvola,threshold_otsu
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.morphology import skeletonize
from skimage.io import imsave
from matplotlib import image
from matplotlib import pyplot as plt
from skimage.filters import unsharp_mask
from skimage.filters.rank import equalize
from skimage.morphology import disk
from skimage.filters import median, threshold_otsu

working_image = imread("/home/pavel/PaperRectification/img/img36.jpg")

canvas = image.imread('/home/pavel/PaperRectification/img/img36.jpg')

height = working_image.shape[0]
width = working_image.shape[1]

gray_image = rgb2gray(working_image)
thresh = threshold_niblack(gray_image)
binary = np.invert(gray_image > thresh)
binary = img_as_ubyte(skeletonize(binary))

imsave("/home/pavel/PaperRectification/img/binary.jpg", binary)

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(binary, theta=tested_angles)


horizontal_lines = []
vertical_lines = []
cross_vertical = []
cross_horizontal = []

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

    a,b = np.cos(angle), np.sin(angle)
    x0, y0 = dist * np.array([a, b])
    pt1 = (x0 + (width)*(-b), y0 + (height)*(a))
    pt2 = (x0 - (width)*(-b), y0 - (height)*(a))

    degress = (angle * 180 / np.pi)
    #plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='r')
    if np.abs(degress) > 0 and np.abs(degress) < 45:
        plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='r')
        cross_vertical.append( np.array([np.cross((pt1[0], pt1[1], 1), (pt2[0], pt2[1], 1)), angle, dist],dtype=object))
    if np.abs(degress) > 70 and np.abs(degress) < 100:
        plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.3, color='b')
        cross_horizontal.append( np.array([np.cross((pt1[0], pt1[1], 1), (pt2[0], pt2[1], 1)), angle, dist],dtype=object))

plt.imshow(canvas)
plt.savefig("/home/pavel/PaperRectification/img/all_lines.jpg", dpi = 1500)

with open('/home/pavel/PaperRectification/npy/cross_vertical.npy', 'wb') as f:
            np.save(f,  cross_vertical)

with open('/home/pavel/PaperRectification/npy/cross_horizontal.npy', 'wb') as f:
            np.save(f, cross_horizontal)

