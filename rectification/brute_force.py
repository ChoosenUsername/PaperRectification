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
from skimage.color import gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean

working_image = imread("img/img1.jpg")

height = working_image.shape[0]
width = working_image.shape[1]

working_image = working_image[height//4:height,0:width]

imsave("halph.jpg", working_image)

gray_image = rgb2gray(working_image)

thresh = threshold_niblack(gray_image)
binary = np.invert(gray_image > thresh)
binary = img_as_ubyte(skeletonize(binary))

imsave("threshold.jpg", binary)

input_image = gray2rgb(binary)
output_image1 = input_image

n = input_image.shape[0] #amount of arrays
m = input_image.shape[1] #amount of columns

vertical_lines = []
lines_sample = []
for x1 in range(m):
    print(x1, "of", m)
    for x2 in range(m):
        rr, cc = line(0, x1, n-1, x2)
        if input_image[rr, cc].mean() > 72:
            output_image1[np.minimum(rr, n-1), np.minimum(cc, m-1)] = [255,0,0]
            vertical_lines.append((rr,cc))
            lines_sample.append(np.cross((x1, 0, 1), (x2, n-1, 1)))

imsave("img_test_72_vert.jpg", output_image1)

first_line = vertical_lines[0]
last_line = vertical_lines[-1]

horizontal_lines = []
lines_sample = []

output_image2 = gray2rgb(binary)


for i in range(first_line[0].shape[0]):

    y1 = first_line[0][i]
    x1 = first_line[1][i]

    for j in range(last_line[0].shape[0]):
        y2 = last_line[0][j]
        x2 = last_line[1][j]

        rr, cc = line(y1, x1, y2, x2)
        if input_image[rr, cc].mean() > 75:
            output_image2[np.minimum(rr, n-1), np.minimum(cc, m-1)] = [255,0,0]
            horizontal_lines.append((rr,cc))
            lines_sample.append(np.cross((x1, y1, 1), (x2, y2, 1)))

imsave("img_test_72_hor.jpg", output_image2)
