import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as F # type: ignore

class HwRModel:
    def __init__(self, model_path, img_w=128, img_h=64):
        self.img_w = img_w
        self.img_h = img_h

        # vocab = 1 spasi + 10 digit + 26 upper + 26 lower
        self.letters = (
            [' '] +
            [str(d) for d in range(10)] +
            [chr(c) for c in range(ord('A'), ord('Z')+1)] +
            [chr(c) for c in range(ord('a'), ord('z')+1)]
        )
        self.VOCAB_SIZE = len(self.letters)
        self.num_classes = self.VOCAB_SIZE + 1  # +1 blank

        # build model
        self.model_pred = self._build_model()
        self.model_pred.load_weights(model_path)

    def _build_model(self):
        input_data = tf.keras.layers.Input(name='input', shape=(self.img_w, self.img_h, 1), dtype='float32')
        # CNN (VGG) layer
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max1')(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max2')(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='max3')(x)

        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', name='conv6', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='max4')(x)

        x = tf.keras.layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Reshape((32, 2048), name='reshape')(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)

        # BiLSTM
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True), name="bilstm1")(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True), name="bilstm2")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # output layer
        x = tf.keras.layers.Dense(self.num_classes, kernel_initializer='he_normal', name='dense2')(x)
        y_pred = tf.keras.layers.Activation('softmax', name='softmax')(x)

        model_pred = tf.keras.Model(inputs=input_data, outputs=y_pred)
        return model_pred

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
        img = self.fix_size(img, self.img_w, self.img_h)

        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: ignore

        img = img.astype(np.float32) / 255.0
        img = img.reshape(self.img_w, self.img_h, 1)
        return np.expand_dims(img, axis=0)  # batch=1

    def decode(self, pred):
        decoded = F.ctc_decode(
            pred,
            input_length=np.ones(pred.shape[0]) * pred.shape[1],
            greedy=True
        )[0][0]

        decoded = F.get_value(decoded)[0]  # array of int
        decoded = decoded[decoded != -1]   # remove padding
        decoded = decoded[decoded < self.VOCAB_SIZE]  # remove blank
        return "".join(self.letters[i] for i in decoded)

    def recognize(self, image_path):
        x = self.preprocess(image_path)
        pred = self.model_pred.predict(x)
        text = self.decode(pred)
        return text
