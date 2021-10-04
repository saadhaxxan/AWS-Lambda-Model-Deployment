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
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from networks.layers import disp_to_depth


def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def load_models(s3, bucket):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    res = ["640x192"]
    response_enc = s3.get_object(Bucket=bucket, Key=f"models/encoder.pth")
    response_dec = s3.get_object(Bucket=bucket, Key=f"models/depth.pth")
    print("Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(
        BytesIO(response_enc["Body"].read()), map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(
        BytesIO(response_dec["Body"].read()), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, feed_height, feed_width


gpu = -1

s3 = boto3.client("s3")
bucket = "depthestimation"

mapping_id_to_style = {0: "640x192"}

encoder, depth_decoder, feed_height, feed_width = load_models(s3, bucket)
print(f"models loaded ...")


def lambda_handler(event, context):
    """
    lambda handler to execute the image transformation
    """
    # warming up the lambda

    data = json.loads(event["body"])
    print("data keys :", data.keys())
    image = data["image"]
    image = image[image.find(",") + 1:]
    dec = base64.b64decode(image + "===")
    image = Image.open(BytesIO(dec))
    image = image.convert("RGB")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    with torch.no_grad():        # Load image and preprocess
        original_width, original_height = image.size
        input_image = image.resize(
            (feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        # output_name = os.path.splitext(os.path.basename(image_path))[0]
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(
            vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[
            :, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        result = {"output": img_to_base64_str(im)}
    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }
