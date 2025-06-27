import cv2
import numpy as np

# Load full image and convert to grayscale

video_path = "/Users/bartlomiejostasz/Downloads/IMG_3566.mov"


image = cv2.imread("Screenshot 2025-06-11 at 01.19.49.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load your checkerboard template (you'll need to crop a small patch manually)
template = cv2.imread("checkerboard_template.png", 0)  # grayscale template
w, h = template.shape[::-1]

# Match template
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

cv2.imshow("Detected Checkerboard", image)
cv2.waitKey(0)
cv2.destroyAllWindows()