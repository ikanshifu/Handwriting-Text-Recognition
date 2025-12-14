import tensorflow as tf
from HwTR import *
import numpy as np
import os

MODEL_PATH = "HwTR_V7.h5"
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