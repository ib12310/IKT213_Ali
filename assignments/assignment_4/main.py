from functions import harris_corner_detection, align_images
import cv2

def main():
    out_dir = "assignments/assignment_4/solutions"
    reference_img=cv2.imread("assignments/assignment_4/reference_img.png")
    align_this = cv2.imread("assignments/assignment_4/align_this.jpg")

    harris_corner_detection(reference_img, out_dir, "harris.png")

    align_images(align_this,
                reference_img, 
                max_features=10,
                good_match_percent=0.7, 
                out_dir=out_dir, 
                aligned_name="aligned.png", 
                matches_name="matches.png")
    

if __name__ == "__main__":
    main()

