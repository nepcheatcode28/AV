# Detect and match image features using OpenCV (ORB)

import cv2
import numpy as np

# Read images in grayscale
img1 = cv2.imread('drama.png', cv2.IMREAD_GRAYSCALE)  # query image
img2 = cv2.imread('drama.png', cv2.IMREAD_GRAYSCALE)  # train image

# Check if images loaded
if img1 is None or img2 is None:
    print("Error: Image not found")
    exit()

# Create ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance (best first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 20 matches
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:60],
    None,
    flags=2
)

# Show result
cv2.imshow("Matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
