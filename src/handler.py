try:
    import unzip_requirements
except ImportError:
    pass

import json
from io import BytesIO
import time
import os
import base64

import boto3
import numpy as np
from PIL import Image

import sys
import PIL.Image as pil
import numpy as np
import tensorflow as tf
import cv2


def preprocess(fp):
    img = cv2.imread(fp)
    img = cv2.resize(img, (513, 513))
    # img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5
    print(img.shape)
    return img


def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def load_models(s3, bucket):
    model = s3.get_object(Bucket=bucket, Key=f"models/decrypted.tflite")
    print("Loading pretrained encoder")
    return model


s3 = boto3.client("s3")
bucket = "skytflite"
model = load_models(s3, bucket)
print(f"models loaded ...")


def lambda_handler(event, context):
    data = json.loads(event["body"])
    print("data keys :", data.keys())
    image = data["image"]
    image = image[image.find(",") + 1:]
    dec = base64.b64decode(image + "===")
    image = Image.open(BytesIO(dec))
    image = image.convert("RGB")

    interpreter = tf.lite.Interpreter(model_path=model)

    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.array(preprocess(image), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    img = output_data
    img = np.squeeze(img)
    img = img[:, :, 0]
    img *= 255.0
    print(img.shape)
    img = np.where(img < 1, 0, img)
    result = {"output": img_to_base64_str(img)}
    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }
