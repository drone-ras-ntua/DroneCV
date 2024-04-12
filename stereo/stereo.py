import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
left_img = cv.imread('images/LEFT.jpg', 0)
left_image = Image.open('images/LEFT.jpg')
right_img = cv.imread('images/RIGHT.jpg', 0)
right_image = Image.open('images/RIGHT.jpg')
#cv.StereoBM
stereo = cv.StereoBM.create(numDisparities=16,  blockSize=5)
disparity = stereo.compute(left_img, right_img)

norm_disparity = cv.normalize(disparity, None, alpha=0, beta=100, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

plt.imshow(disparity)
plt.axis('off')
plt.show()
plt.imshow(norm_disparity, 'gray')
plt.axis('off')
plt.show()