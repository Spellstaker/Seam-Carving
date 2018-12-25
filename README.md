# Seam Carving

## Introduction

This project implements the method described in the paper: <http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf>

It provides image resizing and object removal given a 2d boolean numpy array.

The energy function used is the sum of absolute difference of horizontal and vertical adjacent pixels.

## Usage

Images are represented as 3d numpy arrays. They can be read using various libraries, [OpenCV](https://pypi.org/project/opencv-python/) is used here.

Example: Resizing a 540 x 540 image to 400 x 700
```python
import cv2
from seam import SeamCarve

img = cv2.imread('test.png')
sc_img = SeamCarve(img)
sc_img.rescale(new_height=400, new_width=700)


cv2.imshow('original', img)
cv2.imshow('resized', sc_img.image())
cv2.waitKey(0)
```

![Original](./Example/test.png)

![Resized](./Example/resized.png)