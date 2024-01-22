#!/usr/bin/env python
# coding: utf-8
from io import BytesIO

import numpy as np
import requests
import tflite_runtime.interpreter as tflite
from PIL import Image

# from keras_image_helper import create_preprocessor


# preprocessor = create_preprocessor('xception', target_size=(299, 299))
def load_and_preprocess_image(url):
    # 從 URL 下載圖片
    response = requests.get(url)
    with Image.open(BytesIO(response.content)) as img:
        # 調整圖片尺寸
        img = img.resize((224, 224), Image.NEAREST)

    # 將圖片轉換為 numpy 數組
    image_array = np.array(img, dtype="float32")
    image_array = np.expand_dims(image_array, axis=0)  # 增加批次維度
    # image_array /= 255.0  # 正規化到 [0,1] 範圍

    return image_array


interpreter = tflite.Interpreter(model_path="sport-model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


classes = [
    "air hockey",
    "ampute football",
    "archery",
    "arm wrestling",
    "axe throwing",
    "balance beam",
    "barell racing",
    "baseball",
    "basketball",
    "baton twirling",
    "bike polo",
    "billiards",
    "bmx",
    "bobsled",
    "bowling",
    "boxing",
    "bull riding",
    "bungee jumping",
    "canoe slamon",
    "cheerleading",
    "chuckwagon racing",
    "cricket",
    "croquet",
    "curling",
    "disc golf",
    "fencing",
    "field hockey",
    "figure skating men",
    "figure skating pairs",
    "figure skating women",
    "fly fishing",
    "football",
    "formula 1 racing",
    "frisbee",
    "gaga",
    "giant slalom",
    "golf",
    "hammer throw",
    "hang gliding",
    "harness racing",
    "high jump",
    "hockey",
    "horse jumping",
    "horse racing",
    "horseshoe pitching",
    "hurdles",
    "hydroplane racing",
    "ice climbing",
    "ice yachting",
    "jai alai",
    "javelin",
    "jousting",
    "judo",
    "lacrosse",
    "log rolling",
    "luge",
    "motorcycle racing",
    "mushing",
    "nascar racing",
    "olympic wrestling",
    "parallel bar",
    "pole climbing",
    "pole dancing",
    "pole vault",
    "polo",
    "pommel horse",
    "rings",
    "rock climbing",
    "roller derby",
    "rollerblade racing",
    "rowing",
    "rugby",
    "sailboat racing",
    "shot put",
    "shuffleboard",
    "sidecar racing",
    "ski jumping",
    "sky surfing",
    "skydiving",
    "snow boarding",
    "snowmobile racing",
    "speed skating",
    "steer wrestling",
    "sumo wrestling",
    "surfing",
    "swimming",
    "table tennis",
    "tennis",
    "track bicycle",
    "trapeze",
    "tug of war",
    "ultimate",
    "uneven bars",
    "volleyball",
    "water cycling",
    "water polo",
    "weightlifting",
    "wheelchair basketball",
    "wheelchair racing",
    "wingsuit flying",
]

# url = 'http://bit.ly/mlbookcamp-pants'


def predict(url):
    # X = preprocessor.from_url(url)
    X = load_and_preprocess_image(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
