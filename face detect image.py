# M Samson Badding
import cv2 as cv
from cv2 import imread

# The image file
img = cv.imread('pic1.jpg')

# using the haarcascade frontalface default.xml download from the link below
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
detect = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Converting the image to Grayscale
bnw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_coor = detect.detectMultiScale(bnw)

# The coordinates and detection
for (x, y, w, h) in face_coor:
    rect = cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# Output
cv.imshow('The image', rect)

cv.waitKey()
