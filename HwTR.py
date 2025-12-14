import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as F


class HwTR:
    def __init__(self, img_w, img_h, max_text_length, num_classes, letters, verbose=0):
        self.IMG_W = img_w
        self.IMG_H = img_h
        self.MAX_TEXT_LENGTH = max_text_length
        self.num_classes = num_classes
        self.letters = letters
        self.verbose = verbose
        self.training_model, self.inference_model = self._build_models()

    def _build_models(self):

        input_data = tf.keras.layers.Input(
            name='input', shape=(self.IMG_W, self.IMG_H, 1), dtype='float32'
        )

        # VGG layers
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(input_data)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)

        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)

        x = tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((1,2))(x)

        x = tf.keras.layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((1,2))(x)

        x = tf.keras.layers.MaxPooling2D((2,1))(x)

        x = tf.keras.layers.Conv2D(512, (2,2), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # BiLSTM
        x = tf.keras.layers.Reshape((16, 2048))(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        y_pred = tf.keras.layers.Dense(self.num_classes, kernel_initializer='he_normal')(x)
        y_pred = tf.keras.layers.Activation('softmax', name='softmax')(y_pred)

        # CTC
        labels = tf.keras.layers.Input(name="the_labels", shape=[self.MAX_TEXT_LENGTH], dtype="int32")
        input_length = tf.keras.layers.Input(name="input_length", shape=[1], dtype="int32")
        label_length = tf.keras.layers.Input(name="label_length", shape=[1], dtype="int32")

        # CTC
        loss_out = tf.keras.layers.Lambda(
            lambda args: F.ctc_batch_cost(args[1], args[0], args[2], args[3]),
            name="ctc"
        )([y_pred, labels, input_length, label_length])

        training_model = Model(
            inputs=[input_data, labels, input_length, label_length],
            outputs=loss_out
        )

        inference_model = Model(input_data, y_pred)

        return training_model, inference_model

    def add_padding(self, img, old_w, old_h, new_w, new_h):
        h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
        w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
        img_pad = np.ones([new_h, new_w, 3]) * 255
        img_pad[h1:h2, w1:w2, :] = img
        return img_pad

    def fix_size(self, img, target_w, target_h):
        h, w = img.shape[:2]
        if w < target_w and h < target_h:
            img = self.add_padding(img, w, h, target_w, target_h)
        elif w >= target_w and h < target_h:
            new_w = target_w
            new_h = int(h * new_w / w)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_w, target_h)
        elif w < target_w and h >= target_h:
            new_h = target_h
            new_w = int(w * new_h / h)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_w, target_h)
        else:
            ratio = max(w / target_w, h / target_h)
            new_w = max(min(target_w, int(w / ratio)), 1)
            new_h = max(min(target_h, int(h / ratio)), 1)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_w, target_h)

        return img

    def preprocess(self, path):
        img = cv2.imread(path)
        img = self.fix_size(img, self.IMG_W, self.IMG_H)
    
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        img = img.astype(np.float32) / 255.0
    
        img = img.T                        # Transpose
        img = np.expand_dims(img, axis=-1) # (128, 64, 1)
        img = np.expand_dims(img, axis=0)  # (1, 128, 64, 1)
    
        return img

    def decode(self, preds):
        decoded = F.get_value(
            F.ctc_decode(
                preds,
                input_length=np.ones(preds.shape[0]) * preds.shape[1],
                greedy=True
            )[0][0]
        )

        result = []
        for seq in decoded:
            seq = seq[seq != -1]
            text = "".join(self.letters[i] for i in seq)
            result.append(text)
        return result[0] if len(result) > 0 else ""

    def load(self, checkpoint):
        self.inference_model.load_weights(checkpoint)
        if self.verbose:
            print(f"Weights loaded: {checkpoint}")

    def predict(self, img_tensor):
        preds = self.inference_model.predict(img_tensor)
        return self.decode(preds)

    def preprocess_and_recognize(self, path):
        img = self.preprocess(path)
        return self.predict(img)
