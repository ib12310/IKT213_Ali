import cv2
import os

# Open the default camera
cam = cv2.VideoCapture(0)

# Get camera properties
fps = cam.get(cv2.CAP_PROP_FPS)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Make sure the directory exists
output_dir = os.path.expanduser("~Python/ikt213/assignments/assignment_1/solutions")
os.makedirs(output_dir, exist_ok=True)

# Path for saving the output file
output_file = os.path.join(output_dir, "camera_outputs.txt")

# Save the information into a text file
with open(output_file, "w") as f:
    f.write(f"fps: {fps}\n")
    f.write(f"width: {frame_width}\n")
    f.write(f"height: {frame_height}\n")

# Release the camera
cam.release()
print(f"Camera information saved to {output_file}")
