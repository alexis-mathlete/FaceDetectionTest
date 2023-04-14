import pandas as pd
from ast import literal_eval
import numpy as np
import json
import random

from RunTiledFaces import tile_and_evaluate_tiled
from obj_inference_mosaic import run_image, load_ort
from FaceDetection import MODEL_320, faceDetector

MODEL_320 = True
number_images_to_evaluate = 300

if MODEL_320:
    face_detector_onnx = "./onnx/version-RFB-320.onnx"
    dim = 320
else:
    face_detector_onnx = "./onnx/version-RFB-640.onnx"
    dim = 640

with open('data/fddb_2022_withID.json') as f:
    images_d = json.load(f)

id_l = list(images_d.keys())
url_l = list(images_d.values())
random_index = random.sample(range(0, len(id_l)-1), number_images_to_evaluate)
r_id_l = [id_l[r_int] for r_int in random_index]
r_url_l = [url_l[r_int] for r_int in random_index]


#helper function
def get_num_found(df_entry):
    if pd.isna(df_entry):
        return 0
    elif df_entry == ' ':
        return 0
    else:
        return len(df_entry[2])

def analyze_tiling(image_id_l, url_l, num_tiles_l, image_model= faceDetector, **kwargs):
    #create a summary_df

    output = tile_and_evaluate_tiled(image_id_l, url_l, num_tiles_l, image_model= faceDetector, **kwargs)

    summary_df = pd.DataFrame()
    summary_df['ID'] = output['DataID']
    summary_df['URL'] = url_l
    
    outputy = output.copy()

    #Create list of column names to iterate over when creating accuracy columns
    tiled_results_col_names_l = list(outputy.columns)
    tiled_results_col_names_l.remove('DataID')
    for col_name in tiled_results_col_names_l:
        #add columns to summary data indicating the number of desired objects found in the original and all tiled images
        outputy[col_name+'NumFound'] = outputy[col_name].apply(get_num_found)
    
    #add a column to summary to indicate how many objects were found without tiling
    summary_df['ModelOutputNumFound'] = outputy['ModelOutputNumFound']
    summary_df['PaddedModelOutputNumFound'] = outputy['PaddedModelOutputNumFound']

    tiled_results_col_names_l.remove('ModelOutput')
    tiled_results_col_names_l.remove('PaddedModelOutput')
    
    #report accuracy of finding the desired object in each of the tilings based on the number the model found in the single image
    for col_name in tiled_results_col_names_l:
        #Get the number of tiles (this is the multiplier to get the number of expected objects found)
        multiplier = int(col_name.split('Tile')[0].split('Padded')[0])
        #divide the number of objects found in the tiling by the number annotated times the multiplier
        summary_df[col_name+'Accuracy'] = outputy[col_name+'NumFound']/(multiplier*outputy['ModelOutputNumFound'])

    return summary_df

summary = analyze_tiling(r_id_l, r_url_l, [4,9,16,25])
summary