import os
import cv2
import numpy as np
from numpy.ma.core import empty

from functions import (
    padding, cropping,resizing, copying,
)

BASE = r"C:\Users\ibrah\Python\ikt213\assignments\assignment_2"
BASE_OUT = os.path.join(BASE, "solutions")
os.makedirs(BASE_OUT, exist_ok=True)

img_path = os.path.join(BASE_OUT, "lena-2.png")
image = cv2.imread(img_path)

if image is None:
    raise ValueError("Image could not be loaded")

# 1) Padding with border width 100px
padded = padding(image, border_width=100, out_dir=BASE_OUT)

# 2) Cropping, 80px from top-left, 130 from bottom-right
h, w = image.shape[:2]
x0, y0 = 80, 80
x1, y1 = w - 130, h - 130
cropped = cropping(image, x0, x1, y0, y1, out_dir=BASE_OUT)

# 3) Resizing to 200x200
resized = resizing(image, width=200, height=200, out_dir=BASE_OUT)

# 4) Manually copying into empty array
empty = np.zeros_like(image, dtype=np.uint8)
copied = copying(image, empty, out_dir=BASE_OUT)

