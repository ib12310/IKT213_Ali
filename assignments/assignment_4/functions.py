import cv2
import numpy as np
import os


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(img, out_dir, out_name):
    check_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir, out_name), img)

def harris_corner_detection(
        reference_image, 
        out_dir="assignments/assignment_4/solutions", 
        out_name="harris.png"):

    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

    dst = cv2.dilate(dst, None)

    result = reference_image.copy()
    threshold = 0.01 * dst.max()
    result[dst > threshold] = [0, 0, 255]

    save_image(result, out_dir, out_name)
    return result

def align_images(image_to_align, 
                 reference_image, 
                 max_features,
                 good_match_percent, 
                 out_dir="assignments/assignment_4/solutions", 
                 aligned_name="aligned.png", 
                 matches_name="matches.png"):

    gray1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Could not compute descriptors for one of the images")
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < good_match_percent * n.distance:
            good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda m: m.distance)
    if len(good_matches) < max_features:
        good_matches = good_matches[:max_features]

    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found to compute homography")
    
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography could not be computed")

    height, width = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (width, height))

    mask = mask.ravel().tolist()
    inliner_matches = [m for m, inliner in zip(good_matches, mask) if inliner]

    matches_visualization= cv2.drawMatches(
        image_to_align, keypoints1,
        reference_image, keypoints2, 
        inliner_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    save_image(aligned_image, out_dir, aligned_name)
    save_image(matches_visualization, out_dir, matches_name)

    return aligned_image, matches_visualization