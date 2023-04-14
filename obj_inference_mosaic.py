import os
import sys
import json
import random
from pprint import pprint
from io import BytesIO
import onnxruntime
from PIL import Image
import torchvision.transforms.functional as T
import requests

#from utils import to_numpy
#from dataset import max_size, labels_reference

labels_reference = [
    "background",
    "barcode-qrcode",
    "credit-card",
    "fingerprint",
    "id-card",
    "iris",
    "ssn-card",
    "signature",
    "palmprint",
    "receipt",
    "license-plate",
    "retina",
    "floorplans",
    "chemical-structure",
]

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
min_size = 800
max_size = 1333

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

def load_ort(onnx_model_path):
    _name = onnx_model_path
    _so = onnxruntime.SessionOptions()
    _so.log_severity_level = 3
    _hardware = 'CUDAExecutionProvider'
    ort_session = onnxruntime.InferenceSession(
        _name, _so, providers=[_hardware]
    )
    return ort_session

def run_image(ort_session, image):
    # preprocessing
    background = Image.new("RGBA", image.size, (255, 255, 255))
    img = Image.alpha_composite(background, image).convert("RGB")
    x = to_numpy(T.to_tensor(img.resize((max_size, max_size))).unsqueeze(0))

    ort_inputs = {ort_session.get_inputs()[0].name: x}

    # calculate output
    coords, labels, scores = ort_session.run(None, ort_inputs)

    output = []
    for i in range(len(labels)):
        output.append({
            'prediction': labels_reference[labels[i]],
            'confidence': str(scores[i]),
            'bbox': [str(c) for c in list(coords[i])]
        })
    
    #print(output)
    #with open('test.json', 'w') as f:
    #    json.dump(output, f, indent=4)
    return output

# if __name__ == '__main__':
#     imfile = sys.argv[1]
#     ort_session = load_ort()
#     output = run_image(ort_session, imfile, [0])
#     pprint(output)

# ort_session = load_ort("./onnx/2.1_oos_18oct.onnx")

# run_image(ort_session, 'sample_barcodes/001.jpg', 1)[0]