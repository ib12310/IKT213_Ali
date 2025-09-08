import os
import cv2
import numpy as np

from functions import padding

BASE = r"C:\Users\ibrah\Python\ikt213\assignments\assignment_2"
BASE_OUT = os.path.join(BASE, "solutions")
os.makedirs(BASE_OUT, exist_ok=True)

img_path = os.path.join(BASE_OUT, "lena-2.png")
image = cv2.imread(img_path)

if image is None:
    raise ValueError("Image could not be loaded")

padded = padding(image, border_width=100, out_dir=BASE_OUT)
