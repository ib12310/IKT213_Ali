import os
import cv2
import numpy as np

from functions import padding

BASE = r"assignment_2"
BASE_OUT = os.path.join(BASE, "solutions")
os.makedirs(BASE_OUT, exist_ok=True)

