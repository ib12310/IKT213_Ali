import os
import cv2
import numpy as np
from numpy.ma.core import empty

from functions import (
    padding, crop,resize, copy, grayscale, hsv, hue_shifted, smoothing, rotation
)

BASE = os.path.dirname(os.path.abspath(__file__))

BASE_OUT = os.path.join(BASE, "solutions")
os.makedirs(BASE_OUT, exist_ok=True)

img_path = os.path.join(BASE_OUT, "lena-2.png")
image = cv2.imread(img_path)

if image is None:
    raise ValueError("Image could not be loaded")

# 1) Padding reflecting border
padded = padding(image, border_width=100, out_dir=BASE_OUT)

# 2) Cropping, 80px from top-left, 130 from bottom-right
h, w = image.shape[:2]
x0, y0 = 80, 80
x1, y1 = w - 130, h - 130
cropped = crop(image, x0, x1, y0, y1, out_dir=BASE_OUT)

# 3) Resizing to 200x200
resized = resize(image, width=200, height=200, out_dir=BASE_OUT)

# 4) Manually copying into empty array
empty = np.zeros_like(image, dtype=np.uint8)
copied = copy(image, empty, out_dir=BASE_OUT)

# 5) grayscaling
gray = grayscale(image, out_dir=BASE_OUT)

# 6) Converting to HSV
hsv_image = hsv(image, out_dir=BASE_OUT)

# 7) color shifted by +50 for each channel
empty2 = np.zeros_like(image, dtype=np.uint8)
shifted = hue_shifted(image, empty2, hue=50, out_dir=BASE_OUT)

# 8) Smoothing by blurring
smoothed = smoothing(image, out_dir=BASE_OUT)

# 9) Rotation 90 and 180 degrees
rotate90 = rotation(image, rotation_angle=90, out_dir=BASE_OUT)
rotate180 = rotation(image, rotation_angle=180, out_dir=BASE_OUT)
