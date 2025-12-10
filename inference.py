import tensorflow as tf
from HwR import *
import numpy as np
import os

MODEL_PATH = "HwR.h5"
IMG_H = 64
IMG_W = 128

model = HwRModel(MODEL_PATH, IMG_W, IMG_H)
test_image = "meant.png"

def recognize_img(image):
    text = model.recognize(image)
    return text

print(f"Result : {recognize_img(test_image)}")