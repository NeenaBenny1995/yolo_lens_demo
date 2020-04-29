# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:53:05 2018

@author: Raj Shah
"""
#
import glob
import cv2
from PIL import Image
import os
import sys
import dlib
from skimage import io
from imutils import face_utils
import pandas as pd
import numpy as np

#########################Prediction############################################
# predictor = dlib.shape_predictor("predictor_eye_combine_landmarks.dat")
# detector = dlib.simple_object_detector("detector_eye.svm")
# #detector = dlib.get_frontal_face_detector()
# faces_folder = "/home/raj/Iris Project/BioID/BioID-Faces/"

# ## Now let's run the detector and shape_predictor over the images in the faces
# ## folder and display the results.
# print("Showing detections and predictions on the images in the faces folder...")
# ##win = dlib.image_window()
# for f in glob.glob(faces_folder+"*.png"):
#     print("Processing file: {}".format(f))
#     img = cv2.imread(f)

# ##
# #    #win.clear_overlay()
# #    #win.set_image(img)
# ##
# ##    # Ask the detector to find the bounding boxes of each face. The 1 in the
# ##    # second argument indicates that we should upsample the image 1 time. This
# ##    # will make everything bigger and allow us to detect more faces.
#     dets = detector(img)
#     print("Number of pair of eyes detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
# ##        # Get the landmarks/parts for the face in box d.
# #
#         shape = predictor(img,d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
#         shape = face_utils.shape_to_np(shape)

# ##        # Draw the face landmarks on the screen.
# #    #win.add_overlay(shape)
#     #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
#     einter=np.sqrt((shape[0][0]-shape[1][0])**2+(shape[0][1]-shape[1][1])**2)
#     cv2.circle(img,(shape[0][0],shape[0][1]),1,(255,0,0),-1)
#     cv2.circle(img,(shape[0][0],shape[0][1]),int(0.1*einter),(0,0,255))
#     cv2.circle(img,(shape[1][0],shape[1][1]),1,(255,0,0),-1)
#     cv2.circle(img,(shape[1][0],shape[1][1]),int(0.1*einter),(0,0,255))
#     print(0.1*einter)
#     cv2.imshow("temp",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#################################With Video#######################################

imgEye = cv2.imread("/home/supercom-dev/Desktop/lens1.png",-1)
cv2.imshow("Eye Image 4 channel", imgEye)
h,w,c = imgEye.shape[:3]
# img = np.array(imgEye, dtype=np.float)
# img /= 255.0
# a_channel = np.ones(img.shape, dtype=np.float)/2.0
# image = img*(1-a_channel)

# print( "**", h, w, c)
orig_mask = imgEye[:,:,3]
# cv2.imshow("orig_mask", orig_mask)

orig_mask_inv = cv2.bitwise_not(orig_mask)
# cv2.imshow("orig_mask_inv", orig_mask_inv)

imgEye = imgEye[:,:,0:3]


predictor = dlib.shape_predictor("/home/supercom-dev/Desktop/Iris-Localization-master/predictor_eye_combine_landmarks.dat")
detector = dlib.simple_object_detector("/home/supercom-dev/Desktop/Iris-Localization-master/detector_eye.svm1")
detector2 = dlib.get_frontal_face_detector()

win_det = dlib.image_window()
win_det.set_image(detector)
## Now let's run the detector and shape_predictor over the images in the faces
## folder and display the results.
cap = cv2.VideoCapture(0)
# im_pil = Image.fromarray(cap)
i = 0
while (True):
    ret, frame = cap.read()
    # frame = cv2.imread("/home/ananthu/Desktop/Amar.jpg")
    dets2 = detector2(frame)
    dets = detector(frame)
    print("*******", dets)
    # print("Number of pair of eyes detected: {}".format(len(dets)))
    # print("Number of faces detected: {}".format(len(dets2)))
    if (dets):
        for k, d in enumerate(dets):
            print("kd", k, d)
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
            #                                                                    d.bottom()))
            shape = predictor(frame, d)
            # print("Shape : ", shape)
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
            shape = face_utils.shape_to_np(shape)
            print("Shapeeee : ", shape)

        # cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
        einter = np.sqrt((shape[0][0] - shape[1][0]) ** 2 + (shape[0][1] - shape[1][1]) ** 2)
        # print("einter : ", einter)
        # cv2.circle(frame, (shape[0][0], shape[0][1] + 3), 1, (255, 0, 0), -1)
        # cv2.circle(frame, (shape[0][0], shape[0][1] + 3), int(0.1 * einter), (0, 0, 255))
        # print((shape[0][0], shape[0][1] + 3))
        # cv2.imshow("",frame[(shape[0][1] + 3)-int(0.1 * einter):(shape[0][1] + 3)+int(0.1 * einter), shape[0][0]-int(0.1 * einter):shape[0][0]+int(0.1 * einter)])
        # cv2.circle(frame, (shape[1][0], shape[1][1] + 3), 1, (255, 0, 0), -1)
        # cv2.circle(frame, (shape[1][0], shape[1][1] + 3), int(0.1 * einter), (0, 0, 255))
        # print(0.1 * einter)
        crop_img1 = frame[(shape[0][1] + 3) - int(0.1 * einter):(shape[0][1] + 3) + int(0.1 * einter),
                    shape[0][0] - int(0.1 * einter):shape[0][0] + int(0.1 * einter)]

        x1 = (shape[0][1] + 3) - int(0.1 * einter)

        x2 = (shape[0][1] + 3) + int(0.1 * einter)
        y1 = shape[0][0] - int(0.1 * einter)
        y2 = shape[0][0] + int(0.1 * einter)

        crop_img2 = frame[(shape[1][1] + 3) - int(0.1 * einter):(shape[1][1] + 3) + int(0.1 * einter),
                    shape[1][0] - int(0.1 * einter):shape[1][0] + int(0.1 * einter)]

        xx1 = (shape[1][1] + 3) - int(0.1 * einter)

        xx2 = (shape[1][1] + 3) + int(0.1 * einter)
        yy1 = shape[1][0] - int(0.1 * einter)
        yy2 = shape[1][0] + int(0.1 * einter)
        # cv2.imshow("cropped", crop_img1)
        # cv2.waitKey(0)


        place_eye(crop_img1, crop_img2)


    def place_eye(crop_img1,crop_img2):

        # eyeOverlayHeight=14
        # eyeOverlayWidth=14
        eyeOverlayHeight, eyeOverlayWidth, channels = crop_img1.shape
        # print("sdfsadfadf",eyeOverlayHeight,eyeOverlayWidth)

        # h, w, c = imgEye.shape[:3]
        # print("**", h, w, c)
        eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
        # cv2.imshow("eyeOverlay", eyeOverlay)
        # print("######", eyeOverlay.shape)

        mask = cv2.resize(orig_mask, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)

        roi = frame[x1:x2,y1:y2]
        # cv2.imshow("roi", roi)

        height, width, channel = eyeOverlay.shape[:3]
        # print("roi shape :::: ", roi.shape)
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        cv2.imshow("mask", roi_bg)
        # print("roi_bg shape :::: ", roi_bg.shape)

        roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)
        cv2.imshow("fg", roi_fg)

        dst = cv2.add(roi_bg, roi_fg)
        cv2.imshow("dst", dst)
        #
        frame[(shape[0][1] + 3) - int(0.1 * einter):(shape[0][1] + 3) + int(0.1 * einter),
        shape[0][0] - int(0.1 * einter):shape[0][0] + int(0.1 * einter)] = dst

        ################################################################################################################

        eyeOverlayHeight2, eyeOverlayWidth2, channels2 = crop_img2.shape
        # print("sdfsadfadf", eyeOverlayHeight, eyeOverlayWidth)

        # h, w, c = imgEye.shape[:3]
        # print("**", h, w, c)
        eyeOverlay2 = cv2.resize(imgEye, (eyeOverlayWidth2, eyeOverlayHeight2), interpolation=cv2.INTER_AREA)
        # cv2.imshow("eyeOverlay", eyeOverlay)
        # print("######", eyeOverlay.shape)

        mask2 = cv2.resize(orig_mask, (eyeOverlayWidth2, eyeOverlayHeight2), interpolation=cv2.INTER_AREA)
        mask_inv2 = cv2.resize(orig_mask_inv, (eyeOverlayWidth2, eyeOverlayHeight2), interpolation=cv2.INTER_AREA)

        roi2 = frame[xx1:xx2, yy1:yy2]
        # cv2.imshow("roi", roi)

        height, width, channel = eyeOverlay.shape[:3]
        # print("roi shape :::: ", roi.shape)
        roi_bg2 = cv2.bitwise_and(roi2, roi2, mask=mask_inv2)
        # cv2.imshow("mask", roi_bg)
        # print("roi_bg shape :::: ", roi_bg.shape)

        roi_fg2 = cv2.bitwise_and(eyeOverlay2, eyeOverlay2, mask=mask2)
        # cv2.imshow("fg", roi_fg)

        dst2 = cv2.add(roi_bg2, roi_fg2)
        # cv2.imshow("dst", dst)
        #
        frame[(shape[1][1] + 3) - int(0.1 * einter):(shape[1][1] + 3) + int(0.1 * einter),
        shape[1][0] - int(0.1 * einter):shape[1][0] + int(0.1 * einter)] = dst2

        # cv2.imshow("fra", frame)


    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()


