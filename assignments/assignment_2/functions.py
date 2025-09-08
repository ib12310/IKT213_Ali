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

def cropping(image, x_0, x_1, y_0, y_1, out_dir, out_name="cropped.png"):
    check_dir(out_dir)
    h, w = image.shape[:2]
    x_0, y_0 = max(0, x_0), max(0, y_0)
    x_1, y_1 = min(w, x_1), min(h, y_1)
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite(os.path.join(out_dir, out_name), cropped)
    return cropped

def resizing(image, width, height, out_dir, out_name="resized.png"):
    check_dir(out_dir)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, out_name.format(width, height)), resized)
    return resized

def copying(image, emptyPicArray, out_dir, out_name="copied.png"):
    check_dir(out_dir)
    h, w = image.shape[:2]
    if image.ndim == 2:
        for y in range(h):
            for x in range(w):
                emptyPicArray[y, x] = image[y, x]
    else:
        c = image.shape[2]
        for y in range(h):
            for x in range(w):
                for k in range(c):
                    emptyPicArray[y, x, k] = image[y, x, k]
    cv2.imwrite(os.path.join(out_dir, out_name), emptyPicArray)
    return emptyPicArray


