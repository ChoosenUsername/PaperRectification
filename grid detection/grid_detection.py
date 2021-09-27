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
from skimage import exposure


canvas = image.imread('/home/pavel/PaperRectification/img/all_lines_ar.jpg')

start_image = imread("/home/pavel/PaperRectification/img/all_lines_ar.jpg")
res = start_image


h = start_image.shape[0]
w = start_image.shape[1]

vertical = np.load('/home/pavel/PaperRectification/npy/vertical_debug_lines_ar.npy', allow_pickle = True)
horizontal = np.load('/home/pavel/PaperRectification/npy/horizontal_debug_lines_ar.npy', allow_pickle=True)

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



nums = list(range(40))

dots_on_horizontal_line = np.array(dots_on_horizontal_line)

sorted_indeces_of_hl = np.argsort(dots_on_horizontal_line[:,0])

random_pairs = [ np.random.choice(sorted_indeces_of_hl,2) for i in range(2000)] 
import itertools
for i in random_pairs:
    i1 = i[0]
    i2 = i[1]
    if dots_on_horizontal_line[i1][0] <= dots_on_horizontal_line[i2][0]:
        left_index = i1
        right_index = i2
    else:
        left_index = i2
        right_index = i1

    distance = np.abs( dots_on_horizontal_line[i1][0] - dots_on_horizontal_line[i2][0] )

    left_border = np.where( sorted_indeces_of_hl == left_index)
    right_border = np.where( sorted_indeces_of_hl == right_index)
    left_array = sorted_indeces_of_hl[:left_border[0][0]]
    right_array = sorted_indeces_of_hl[right_border[0][0]:]
    
    