import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys




# ten anh
# fileIMG = sys.argv[1]
fileIMG = "DataImg/Cay hoa song doi/IMG_20220810_134109.jpg"

# img = cv2.imread(fileIMG, cv2.IMREAD_GRAYSCALE) # che do mau xam
img = cv2.imread(fileIMG)

img = cv2.resize(img, (540, 540))     #chinh lai kich thuoc anh

cv2.imshow('Anh La Cay',img) # dat tieu de anh

cv2.waitKey(0)
cv2.destroyAllWindows()