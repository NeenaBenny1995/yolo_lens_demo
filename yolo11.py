# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
import cv2
from timeit import default_timer as timer

import dlib
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
face_cascade = cv2.CascadeClassifier('/home/supercom-dev/Desktop/keras-yolo3-master_TLC/haar/haarcascade_frontalface_default.xml')
detector2 = dlib.get_frontal_face_detector()
imgEye = cv2.imread("/home/supercom-dev/Desktop/keras-yolo3-master_TLC/eye1.png",-1)
# cv2.imshow("Eye Image 4 channel", imgEye)
h,w,c = imgEye.shape[:3]

# print( "**", h, w, c)
orig_mask = imgEye[:,:,3]
# cv2.imshow("orig_mask", orig_mask)

orig_mask_inv = cv2.bitwise_not(orig_mask)
# cv2.imshow("orig_mask_inv", orig_mask_inv)

imgEye = imgEye[:,:,0:3]



class YOLO(object):
    _defaults = {
        "model_path": '/home/supercom-dev/Desktop/keras-yolo3-master_TLC/logs/000/trained_weights_final.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'eye.txt',
        "score" : 0.99,
        "iou" : 0.45,
        "model_image_size" : (640,1280),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,frame,x,y):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            # print("before",top, left, bottom, right)
            # print(image.size[1],image.size[0])
            # top = int(max(0, np.floor(top + 0.5)))
            # print("top--",top)
            # left = int(max(0, np.floor(left + 0.5)))
            # print("left--", left)
            # bottom = int(min(image.size[1], np.floor(bottom + 0.5)))
            # print("bottom--", bottom)
            # right = int(min(image.size[0], np.floor(right + 0.5)))
            # print("right--", right)
            top = max(0, np.floor(top + 0.5))
            left = max(0, np.floor(left + 0.5))
            bottom = min(image.size[1], np.floor(bottom + 0.5))
            right = min(image.size[0], np.floor(right + 0.5))


            print(label, (left, top), (right, bottom))
            # images = np.asarray(image)
            # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            # cv2.imshow("", images)
            # crop1=cv2.getRectSubPix(image, (top,bottom),(left,right))
            # cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            crop_img1 = frame[int(top+y):int(bottom+y), int(left+x):int(right+x)]
#
#
# ######################################################################################
            eyeOverlayHeight, eyeOverlayWidth, channels = crop_img1.shape
            # print("sdfsadfadf",eyeOverlayHeight,eyeOverlayWidth)

            # h, w, c = imgEye.shape[:3]
            # print("**", h, w, c)

            eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
            # cv2.imshow("eyeOverlay", crop_img1)
            # print("######", eyeOverlay.shape)

            mask = cv2.resize(orig_mask, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("mask", mask)
            # cv2.imshow("mask_inv", mask_inv)
            roi = frame[int(top+y):int(bottom+y), int(left+x):int(right+x)]
            # cv2.imshow("mask", roi)
            face_part = (roi * (1 / 255.0)) * (mask_inv * (1 / 255.0))
            # cv2.imshow("mask", face_part)
            overlay_part = (eyeOverlay * (1 / 255.0)) * (mask * (1 / 255.0))
            # cv2.imshow("roi", roi)
            # roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            # cv2.imshow("bg", roi_bg)
            # print("roi_bg shape :::: ", roi_bg.shape)

            # roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)
            dst = cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0)
            # cv2.imshow("fg", roi_fg)

            # dst = cv2.add(roi_bg, roi_fg)
            # cv2.imshow("dst", dst)
            # cv2.rectangle(images,(left+x,top+y), (right+x,bottom+y),(255, 0, 0),2)
            frame[int(top+y):int(bottom+y), int(left+x):int(right+x)] = dst
            # cv2.imshow("--", frame)







########################################################################################




            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])
            #
            # # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # del draw

        # print()
        end = timer()
        print(end - start)
        return frame

    def close_session(self):
        self.sess.close()

def detect_video(yolo):
    import cv2
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    # video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    # video_fps       = vid.get(cv2.CAP_PROP_FPS)
    # video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                     int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # isOutput = True if output_path != "" else False
    # if isOutput:
    #     print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    #     out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        # frame = cv2.imread("/home/supercom-dev/Pictures/ibug_300W_large_face_landmark_dataset/afw/5020454673_1_mirror.jpg")
        frame1=frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        if faces != ():
            x = faces[0][0]
            y = faces[0][1]
            w = faces[0][2]
            h = faces[0][3]
            roi_color = frame[y:y + h, x:x + w]
            image = Image.fromarray(roi_color)
            images = yolo.detect_image(image,frame,x,y)
            # frame1=images.copy()
            # result = np.asarray(image)
            # curr_time = timer()
            # exec_time = curr_time - prev_time
            # prev_time = curr_time
            # accum_time = accum_time + exec_time
            # curr_fps = curr_fps + 1
            # if accum_time > 1:
            #     accum_time = accum_time - 1
            #     fps = "FPS: " + str(curr_fps)
            #     curr_fps = 0
            #

            # cv2.imshow("imf",images)
            # cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.50, color=(255, 0, 0), thickness=2)
            # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            img_gray1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            mask1 = np.zeros_like(img_gray1)
            # detector1 = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(
                "/home/supercom-dev/Desktop/keras-yolo3-master_TLC/shape_predictor_81_face_landmarks.dat")
            faces1 = detector2(img_gray1)

            index_dst = []
            for face1 in faces1:
                landmarks1 = predictor(img_gray1, face1)
                landmarks_points1 = []
                for n1 in range(36, 48):
                    x = landmarks1.part(n1).x
                    y = landmarks1.part(n1).y
                    landmarks_points1.append((x, y))
                    np.array(landmarks_points1)
                    face_points1 = []
                    points1 = np.array(landmarks_points1, np.int32)

                #
                convexhull1 = cv2.convexHull(points1)
                # cv2.polylines(img, [points], True, (255, 0, 0), 3)
                cv2.fillConvexPoly(mask1, convexhull1, 255)
                fg = cv2.bitwise_or(images, images, mask=mask1)
                mask_inv = cv2.bitwise_not(mask1)
                bk = cv2.bitwise_or(frame1, frame1, mask=mask_inv)
                final = cv2.bitwise_or(fg, bk)
                cv2.imshow("out.jpg", final)




            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # else:
        #     cv2.imshow("frame",frame)
    # yolo.close_session()


if __name__ == '__main__':

    frnt = YOLO()
    detect_video(frnt)
