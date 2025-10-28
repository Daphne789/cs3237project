import cv2
import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('apriltag_train_data.csv')
images, labels = [], []


for i, row in df.iterrows():
    img_path = row["img_filepath"]
    label = 1 if row["has_apriltag"] else 0

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    images.append(img)
    labels.append(label)

print("Images:", images)
print("Labels:", labels)
