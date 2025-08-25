import cv2

# Load a color image in grayscale
lena = cv2.imread('lena-1.png')

def print_image_information(image):
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)

    print("Size:", image.size)
    print("Data type:", image.dtype)

print_image_information(lena)