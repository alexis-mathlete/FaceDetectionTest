import os
import sys
from pprint import pprint
import cv2
import json
import onnxruntime as ort
import argparse
import numpy as np
from PIL import Image
from resize import pad_to_square

from box_utils import predict


# select either the 640x480 or 320x240 model. This determines the resize shape.
MODEL_320 = False 

if MODEL_320:
    face_detector_onnx = "./onnx/version-RFB-320.onnx"
else:
    face_detector_onnx = "./onnx/version-RFB-640.onnx"

face_detector = ort.InferenceSession(face_detector_onnx)

def faceDetector(image, pad = False, threshold = 0.7):
    """
    function to preprocess input image and run onnx inference
    """
    # this is just a color space conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if pad:
        image = pad_to_square(image)
    # resize the model to either 320x240 or 640x480
    if MODEL_320:
        image = cv2.resize(image, (320, 240))
    else:
        image = cv2.resize(image, (640, 480))

    # subtract mean pixel value from image and then scale
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128

    # reshape 
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    # run onnx model
    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})

    # process the predictions
    # the predict() function does non-maximal suppression so that we don't get multiple boxes
    # over a single face
    boxes, labels, probs = predict(image.shape[1],
                                   image.shape[0],
                                   confidences,
                                   boxes,
                                   threshold)

    return boxes, labels, probs