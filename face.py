import cv2 as cv
from cv2 import imread

# img = cv.imread("multiface1.jpg")
img = cv.imread("JhonnyDepp.jpg")
# half = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1)

# cv.imshow('faces',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray faces' , gray)

haar_cascade = cv.CascadeClassifier('detector.xml')
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors =5)
print(f'total faces found : {len(face_rect)}')

for (x,y,i,j) in face_rect:
    cv.rectangle(img, (x,y), (x+i,y+j), (255,0,0), thickness=2)

cv.imshow("detected faces", img)

cv.waitKey(0)