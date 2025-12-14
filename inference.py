import tensorflow as tf
from HwTR import *
import numpy as np
import os

MODEL_PATH = "HwTR_V9.h5"
IMG_H = 64
IMG_W = 128
LETTERS = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)

hwr = HwTR(
    img_w=128,
    img_h=64,
    max_text_length=16,
    num_classes=len(LETTERS)+1,
    letters=LETTERS
)

hwr.load(MODEL_PATH)
test_image = "demo.png"

print(f"Result : {hwr.preprocess_and_recognize(test_image)}")

import tensorflow as tf

model = tf.keras.models.load_model("HwTR_V9.h5", custom_objects={"CTCLayer": CTCLayer})

image_input = model.get_layer("input").input
output_layer = model.get_layer("softmax").output
inference_model = tf.keras.models.Model(image_input, output_layer)

inference_model.save("HwTR_V9_inference.h5")

print("Success! Download 'HwTR_V9_inference.h5' and upload it to GitHub.")