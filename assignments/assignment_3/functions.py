import cv2
import numpy as np
import os


def check_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_image(img, out_dir, out_name):
    check_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir, out_name), img)

def sobel_edge_detection(image, out_dir, out_name="sobel_edge_detection.png"):

    # Sobel on a blurred grayscale image (dx=1, dy=1, ksize=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_64f = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)
    sobel_abs = cv2.convertScaleAbs(sobel_64f)
    cv2.imwrite(os.path.join(out_dir, out_name), sobel_abs)
    save_image(sobel_abs, out_dir, out_name)
    return sobel_abs


def canny_edge_detection(image, threshold_1, threshold_2, out_dir, out_name="canny.png"):

    # Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    save_image(edges, out_dir, out_name)    
    return edges


def template_match(image, template, out_dir, out_name="template_match.png"):
    
    # Template match with multiple detections
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template.copy()
    w, h = tpl_gray.shape[1], tpl_gray.shape[0]
    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.9)
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(vis, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    save_image(vis, out_dir, out_name)
    return vis


def resize(image, scale_factor: int, up_or_down: str, out_dir, out_name="resized_{}_x{}.png"):
    
    # Resizing (image pyramids)
    steps = 1 if scale_factor == 2 else max(1, int(np.log2(max(2, scale_factor))))

    resized_img = image.copy()
    for _ in range(steps):
        if up_or_down == "up":
            resized_img = cv2.pyrUp(resized_img)
        else:
            resized_img = cv2.pyrDown(resized_img)
    save_image(resized_img, out_dir, out_name.format(up_or_down, scale_factor))
    return resized_img
