import cv2
import numpy as np
import os

def check_dir(path: str):
    os.makedirs(path, exist_ok=True)

def padding(image, border_width, out_dir, out_name="padded.png"):
    check_dir(out_dir)
    padded = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_CONSTANT,
    )
    cv2.imwrite(os.path.join(out_dir, out_name), padded)
    return padded

def cropping(image, x0, x1, y0, y1, out_dir, out_name="cropped.png"):
    check_dir(out_dir)
    h, w = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    cropped = image[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(out_dir, out_name), cropped)
    return cropped
