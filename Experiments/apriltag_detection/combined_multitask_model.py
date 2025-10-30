import numpy as np
import pandas as pd
import cv2
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv('apriltag_train_data.csv')

IMG_W = 64
IMG_H = 64
images = []
cls_labels = [] 
corners_all = []

for _, row in df.iterrows():
    img_path = row['img_filepath']
    has_tag = bool(row['has_apriltag'])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0

    cls_labels.append(1 if has_tag else 0)

    if has_tag:
        c = np.array([float(x) for x in row['corners'].split(',')])
        c[0::2] /= 240.0
        c[1::2] /= 240.0
    else:
        c = np.zeros(8, dtype=np.float32)

    images.append(img)
    corners_all.append(c)

X = np.expand_dims(np.array(images), -1)
y_cls = np.array(cls_labels)
y_reg = np.array(corners_all)

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(X, y_cls, y_reg, test_size=0.2, random_state=42)

inputs = keras.Input(shape=(64, 64, 1))

# shared cnn layers will be used for both clf and reg
x = layers.Conv2D(16, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(32, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)

cls_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)
reg_output = layers.Dense(8, activation='linear', name='corner_output')(x)

idx_tag = np.where(y_cls == 1)[0][0]
idx_notag = np.where(y_cls == 0)[0][0]
plt.subplot(1,2,1); plt.imshow(X[idx_tag].squeeze(), cmap='gray'); plt.title("has_tag=1")
plt.subplot(1,2,2); plt.imshow(X[idx_notag].squeeze(), cmap='gray'); plt.title("has_tag=0")
plt.show()


model = models.Model(inputs=inputs, outputs=[cls_output, reg_output])

model.compile(
    optimizer='adam',
    loss={
        'class_output': 'binary_crossentropy',
        'corner_output': 'mse'
    },
    loss_weights={
        'class_output': 1.0,
        'corner_output': 5.0 
    },
    metrics={
        'class_output': ['accuracy'],
        'corner_output': ['mae']
    }
)

history = model.fit(
    X_train,
    {'class_output': y_cls_train, 'corner_output': y_reg_train},
    epochs=60,
    batch_size=16,
    validation_split=0.1
)

eval_results = model.evaluate(X_test, {'class_output': y_cls_test, 'corner_output': y_reg_test})
print("Clf loss:", eval_results[1])
print("Reg loss (MSE):", eval_results[2])
print("clf accuracy percentage:", eval_results[3]*100)
print("reg mae:", eval_results[4])

model.save("apriltag_multitask.keras")

# convert to tflite quantised
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open("apriltag_multitask_quant.tflite", "wb").write(tflite_model)
print("Saved quantized model to apriltag_multitask_quant.tflite")
