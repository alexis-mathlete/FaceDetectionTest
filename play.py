from torch import tile
from imageio import cv2_to_pil
import os
from obj_inference_mosaic import run_image, load_ort
import pandas as pd
import math
from mosaic import concat_square
import numpy as np
from imageio import save_cv2, loads_pil
import requests
from PIL import Image
import torchvision.transforms.functional as T
from io import BytesIO
import cv2
from imageio import pil_to_cv2
from resize import pad_to_square
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import ast
import validators
import json
from FaceDetection import MODEL_320, faceDetector

df = pd.read_csv('Data/vgg_face_data.csv')
with open('Data/vgg_face_data.json', 'w') as f:
    json.dump(df,f)

url_l = df['Labeled Data'].tolist()
id_l = df['ID'].tolist()


with open('./Data/fddb_2022.json') as f:
    url_l = json.load(f)

num_files = len(url_l)
fddb_2022_d = dict(zip(list(range(num_files)), url_l))
with open('data/fddb_2022_withID.json', "w") as outfile:
    json.dump(fddb_2022_d,outfile)

valid=validators.url(url_l[0])

response = requests.get('https://skysync-act-public.s3.us-east-2.amazonaws.com/images/_fddb_2002/000c0e9d-0998-40ef-a5c2-e7d0b49c0fde.jpg')
png = Image.open(BytesIO(response.content)).convert('RGBA')
im = pil_to_cv2(png)
padded_im = pad_to_square(im)

output = faceDetector(padded_im)
output

output

