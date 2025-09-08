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
