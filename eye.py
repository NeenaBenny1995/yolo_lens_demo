import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('/home/supercom-dev/Desktop/keras-yolo3-master_TLC/haar/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('/home/supercom-dev/Desktop/keras-yolo3-master_TLC/haar/haarcascade_eye.xml')

# cap = cv2.VideoCapture(0)
img=cv2.imread("/home/supercom-dev/Desktop/keras-yolo3-master_TLC/abcd.jpg")
# while True:
#     ret, img = cap.read()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print(faces)
x=faces[0][0]
y=faces[0][1]
w=faces[0][2]
h=faces[0][3]

cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
roi_gray = gray[y:y + h, x:x + w]
roi_color = img[y:y + h, x:x + w]

eyes = eye_cascade.detectMultiScale(roi_gray)
print(eyes)
ww=hh=0
if eyes[0][0]<eyes[1][0]:
    minx=eyes[0][0]
    ww=eyes[1][2]
    hh=eyes[1][3]
else:
    minx = eyes[1][0]
    ww = eyes[0][2]
    hh = eyes[0][3]
# minx=min(eyes[0][0],eyes[1][0])
miny=min(eyes[0][1],eyes[1][1])
maxx=max(eyes[0][0],eyes[1][0])
maxy=max(eyes[0][1],eyes[1][1])
# height=max(eyes[0][2],eyes[1][2])
# width=max(eyes[0][3],eyes[1][3])
print(minx,miny,maxx,maxy)
print(minx+x,miny+y)
print(maxx+x+ww,maxy+y+hh)
top=(minx+x,miny+y)
bot=(maxx+x+ww,maxy+y+hh)
# cv2.rectangle(img, top, bot, (0, 255, 0), 2)
# cv2.imshow('img', img)
cv2.imshow("iii",img[miny+y:maxy+y+hh,minx+x:maxx+x+ww])


#
# cv2.imshow('img', img)
cv2.waitKey(0)




# cap.release()
cv2.destroyAllWindows()