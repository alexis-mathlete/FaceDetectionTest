from torch import tile
from yaml import load
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
from FaceDetection import MODEL_320, faceDetector
import json
import random
#Load model
model1 = load_ort('onnx/version-RFB-320.onnx')
model2 = load_ort('onnx/version-RFB-640.onnx')

MODEL_320 = True

if MODEL_320:
    face_detector_onnx = "./onnx/version-RFB-320.onnx"
    dim = 320
else:
    face_detector_onnx = "./onnx/version-RFB-640.onnx"
    dim = 640

def tile_and_evaluate_tiled(image_id_l, image_url_l, num_tiles_l, image_model= faceDetector, **kwargs):
    df_column_names = [f'{n}TileOutput' for n in num_tiles_l]
    df_padded_columns_names = [f'{n}PaddedTileOutput' for n in num_tiles_l]
    df_column_names = ['DataID', 'ModelOutput', 'PaddedModelOutput'] + df_column_names + df_padded_columns_names
    df_output = pd.DataFrame(columns = df_column_names)
    num_files = len(image_url_l)
    for i, data_url in enumerate(image_url_l):
        print(f'Processed {i} of {num_files} images.')
        #Get the image
        valid=validators.url(data_url)
        if valid == False:
            continue
        response = requests.get(data_url)
        try:
            png = Image.open(BytesIO(response.content)).convert('RGBA')
            im = pil_to_cv2(png)
        except:
            continue
        #Run the model on the original image

        #pad the image to a square
        padded_im = pad_to_square(im)

        #Run the model on both the original and the padded image
        model_output = image_model(image = im,**kwargs)
        padded_model_output = image_model(image = padded_im)
        
        #If an object is detected in the original image, tile the image
        if (len(model_output[1]) != 0) or (len(padded_model_output[1]) != 0):
            
            #form list of tiled images (cv2 type) for each num_tiles (4,9,16) in num_tiles_l
            image_l_l = [[im]*n for n in num_tiles_l]
            padded_image_l_l = [[padded_im]*n for n in num_tiles_l]
            all_images_l_l = image_l_l + padded_image_l_l
            tiled_image_l = [concat_square(image_l, dim) for image_l in all_images_l_l]
            # #convert list of tiled images to list of pil images
            # tiled_pil_l = [cv2_to_pil(tiled_image).convert('RGBA') for tiled_image in tiled_image_l]
            #run the model on each tiled image 4,9,16...
            model_output_l = [image_model(image = tiled_im, **kwargs) for tiled_im in tiled_image_l]
            #create row to append to dataframe
            new_df_row_l = [image_id_l[i], model_output, padded_model_output] + model_output_l
            new_df_row_d = dict(zip(df_column_names, new_df_row_l))
            df_output = df_output.append(new_df_row_d, ignore_index= True)
        #if an object is not detected in the original image, add empty results
        else:
            outputs = [image_id_l[i], ' ', ' '] + [' ']*len(num_tiles_l)
            new_df_row_d = dict(zip(df_column_names, outputs))
            df_output = df_output.append(new_df_row_d, ignore_index= True)
    return df_output

# output = tile_and_evaluate_tiled(r_id_l, r_url_l, [4, 9, 16])

# df = pd.read_json(barcode_json2)
# id_l = df['ID'].tolist()[:5]
# url_l = df['Labeled Data'].tolist()[:5]

# output = tile_and_evaluate_tiled(id_l, url_l, [4,9], image_model = run_image, ort_session = ort_session)
