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
        cv2.BORDER_REFLECT_101
    )
    cv2.imwrite(os.path.join(out_dir, out_name), padded)
    return padded


def crop(image, x_0, x_1, y_0, y_1, out_dir, out_name="cropped.png"):
    check_dir(out_dir)
    h, w = image.shape[:2]
    x_0, y_0 = max(0, x_0), max(0, y_0)
    x_1, y_1 = min(w, x_1), min(h, y_1)
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite(os.path.join(out_dir, out_name), cropped)
    return cropped


def resize(image, width, height, out_dir, out_name="resized.png"):
    check_dir(out_dir)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, out_name.format(width, height)), resized)
    return resized


def copy(image, emptyPicArray, out_dir, out_name="copied.png"):
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


def grayscale(image, out_dir, out_name="grayscale.png"):
    check_dir(out_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, out_name), gray)
    return gray


def hsv(image, out_dir, out_name="hsv.png"):
    check_dir(out_dir)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(out_dir, out_name), hsv_img)
    return hsv_img


def hue_shifted(image, emptyPicArray, hue, out_dir, out_name="hue_shift.png"):
    check_dir(out_dir)
    add_scalar = (hue, hue, hue)
    shifted = cv2.add(image, np.array(add_scalar, dtype=np.uint8))
    emptyPicArray[:] = shifted
    cv2.imwrite(os.path.join(out_dir, out_name.format(hue)), shifted)
    return shifted

def smoothing(image, out_dir, out_name="smoothed.png"):
    check_dir(out_dir)
    blurred = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite(os.path.join(out_dir, out_name), blurred)
    return blurred

def rotation(image, rotation_angle, out_dir, out_name="rotated_{}.png"):
    check_dir(out_dir)
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(out_dir, out_name.format(rotation_angle)), rotated)
    return rotated