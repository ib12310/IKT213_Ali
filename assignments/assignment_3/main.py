import os
import cv2
import numpy as np

from functions import (
    check_dir, sobel_edge_detection,
    canny_edge_detection, template_match,
    resize,
)

# paths
BASE = os.path.dirname(os.path.abspath(__file__))
BASE_OUT = os.path.join(BASE, "solutions")
os.makedirs(BASE_OUT, exist_ok=True)


# lambo image
img_path = os.path.join(BASE, "lambo.png")
image = cv2.imread(img_path)
if image is None:
    raise ValueError(f"Image not found in: {img_path}")

# template matching images
shapes_path = os.path.join(BASE, "shapes-1.png")
template_path = os.path.join(BASE, "shapes_template.jpg")
shapes_img = cv2.imread(shapes_path)
template_img = cv2.imread(template_path)
if shapes_img is None:
    raise ValueError(f"shapes png not found in: {shapes_path}")
if template_img is None:
    raise ValueError(f"'shapes_template.jpg' not found in: {template_path}")

# Sobel
sobel_edges = sobel_edge_detection(image, out_dir=BASE_OUT)

# Canny 
canny_edges = canny_edge_detection(image, threshold_1=50, threshold_2=50, out_dir=BASE_OUT)

# Template match 
matched = template_match(shapes_img, template_img, out_dir=BASE_OUT)

# Resizing with pyramids 
resized_up = resize(image, scale_factor=2, up_or_down="up", out_dir=BASE_OUT, out_name="resized_{}_x{}.png")
resized_down = resize(image, scale_factor=2, up_or_down="down", out_dir=BASE_OUT, out_name="resized_{}_x{}.png")
