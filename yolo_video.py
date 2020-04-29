
import cv2
import yolo
from PIL import Image


FLAGS = None
face_cascade = cv2.CascadeClassifier(r'C:\Users\user\Pictures\work from home\keras-yolo3-master_TLC\haar\haarcascade_frontalface_default.xml')
    # # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier(r'C:\Users\user\Pictures\work from home\keras-yolo3-master_TLC\haar\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
if __name__ == '__main__':
    import numpy as np


    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

    #
    while True:
        _,img =cap.read()
        print(img)
        # # while True:
        # #     ret, img = cap.read()
        #
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        while faces!=():
            x = faces[0][0]
            y = faces[0][1]
            w = faces[0][2]
            h = faces[0][3]
            roi_color = img[y:y + h, x:x + w]
            #
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # print(eyes)
            # ww = hh = 0
            # if eyes[0][0] < eyes[1][0]:
            #     minx = eyes[0][0]
            #     ww = eyes[1][2]
            #     hh = eyes[1][3]
            # else:
            #     minx = eyes[1][0]
            #     ww = eyes[0][2]
            #     hh = eyes[0][3]
            # # minx=min(eyes[0][0],eyes[1][0])
            # miny = min(eyes[0][1], eyes[1][1])
            # maxx = max(eyes[0][0], eyes[1][0])
            # maxy = max(eyes[0][1], eyes[1][1])
            # # height=max(eyes[0][2],eyes[1][2])
            # # width=max(eyes[0][3],eyes[1][3])
            # print(minx, miny, maxx, maxy)
            # top = (minx + x, miny + y)
            # bot = (maxx + x + ww, maxy + y + hh)
            # # # image=Image.open("/home/supercom-dev/Desktop/keras-yolo3-master_TLC/abcd.jpg")
            # roi_og=img[miny+y:maxy+y+hh,minx+x:maxx+x+ww]
            # #
            # # # im_pil.show()
            # dim = (640, 1280)
            # # # resize image
            # img = cv2.imread("/home/supercom-dev/Desktop/keras-yolo3-master_TLC/abcd.jpg")
            # resized = cv2.resize(roi_og, dim, interpolation=cv2.INTER_AREA)
            im_pil = Image.fromarray(roi_color)
            # cv2.imshow("iii", roi_og)
            #
            # #
            # # cv2.imshow('img', img)
            # cv2.waitKey(0)
            #
            # # cap.release()
            # cv2.destroyAllWindows()
            frnt = yolo.YOLO()
            r_image = frnt.detect_image(im_pil)
            r_image.show()
            if ord("q"):
                break
