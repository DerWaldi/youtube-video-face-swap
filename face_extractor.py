import cv2, os, glob, tqdm, sys
import numpy

from utils import get_image_paths, load_images, stack_images

import dlib
import face_recognition
import face_recognition_models
from umeyama import umeyama

from moviepy.editor import *
from tqdm import tqdm

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_68_point_model)

mean_face_x = numpy.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = numpy.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = numpy.stack( [ mean_face_x, mean_face_y ], axis=1 )

# get face align matrix from landmarks
def get_align_mat(face):
    return umeyama( numpy.array(face.landmarksAsXY()[17:]), landmarks_2D, True )[0:2]

# get inverse face align matrix from landmarks
def get_align_mat_inv(face):
    return umeyama(landmarks_2D, numpy.array(face.landmarksAsXY()[17:]), True )[0:2]

# detect faces in image
def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    landmarks = _raw_face_landmarks(frame, face_locations)   

    for ((y, right, bottom, x), landmarks) in zip(face_locations, landmarks):
        yield DetectedFace(frame[y: bottom, x: right], x, right - x, y, bottom - y, landmarks)

# extract all faces in image
def extract_faces(image, size):    
    facelist = []
    for face in detect_faces(image):
        if face.landmarks == None:
            print("Warning! landmarks not found. Switching to crop!")
            facelist.append((face, cv2.resize(face.image, (size, size))))

        alignment = get_align_mat( face )
        facelist.append((face, transform( image, alignment, size, 48 )))
    return facelist
    
def transform(image, mat, size, padding=48 ):
    mat = mat * (size - 2 * padding)
    mat[:,2] += padding
    return cv2.warpAffine( image, mat, ( size, size ) )

def _raw_face_landmarks(face_image, face_locations):
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

# detected face class
class DetectedFace(object):
    def __init__(self, image, x, w, y, h, landmarks):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarks = landmarks

    # retrieve landmarks as tuple list
    def landmarksAsXY(self):
        return [(p.x, p.y) for p in self.landmarks.parts()]

# this method is used to insert extracted faces again into the original image
def blend_warp(src, dst, mat):
    # use some kind of blend to smooth the border
    imgMask = numpy.ones(src.shape)
    imgMask[0,:,:] = 0
    imgMask[:,0,:] = 0
    imgMask[-1,:,:] = 0
    imgMask[:,-1,:] = 0
    imgMaskWarped = cv2.warpAffine( imgMask, mat, (dst.shape[1],dst.shape[0]))[:, :, 0]
    
    src_warped = cv2.warpAffine( src, mat, (dst.shape[1],dst.shape[0]))
    # make the colors smoother with a maximum face alpha of 95%
    alpha = imgMaskWarped * 0.95
    beta = (1.0 - alpha)
    res_warped = dst.copy()
    for c in range(0, 3):
        res_warped[:, :, c] = (beta * dst[:, :, c] + alpha * src_warped[:, :, c])
    
    return res_warped