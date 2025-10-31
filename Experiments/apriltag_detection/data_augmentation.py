import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import albumentations as A
import os

AUGMENTED_OUTPUT_CSV = "augmented_apriltag_train_data.csv"
AUGMENTED_OUTPUT_DIR = "captured_frames"
NUM_AUGS_PER_IMAGE = 4

def read_image_for_augmentation(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    img = np.expand_dims(img, axis=-1)  # make shape (H, W, 1)
    return img

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
        shear=(-5, 5),
        p=0.8
    ),
    A.HorizontalFlip(p=0.5),
],
keypoint_params=A.KeypointParams(
    format='xy', 
    remove_invisible=False
))

def augment_and_save_images(df_ranged, output_dir=AUGMENTED_OUTPUT_DIR, num_augs=NUM_AUGS_PER_IMAGE):
    os.makedirs(output_dir, exist_ok=True)

    new_rows = []
    counter = 0

    for _, row in df_ranged.iterrows():
        img_filepath = row['img_filepath']
        has_apriltag = bool(row['has_apriltag'])
        if not has_apriltag: continue
        corners_col = row['corners']
        corners = np.array([float(x) for x in corners_col.split(',')], dtype=np.float32)
        img = read_image_for_augmentation(img_filepath)
        h, w = img.shape[:2]
        c = corners.reshape(-1, 2)

        for _ in range(num_augs):
            augmented = augment(image=img, keypoints=c)
            aug_img = augmented['image']
            aug_corners = np.array(augmented['keypoints'])

            if aug_img.ndim == 3 and aug_img.shape[2] == 1:
                aug_img = aug_img.squeeze(-1)

            new_filename = f"frame_{1000000000 + counter}.jpg"
            save_path = os.path.join(output_dir, new_filename)

            cv2.imwrite(save_path, aug_img)
            counter += 1

            aug_corners_flat = aug_corners.flatten().astype(np.float32)
            new_rows.append({
                "img_filepath": save_path,
                "has_apriltag": True,
                "corners": ",".join(map(str, aug_corners_flat))
            })

            print(f"Saved: {save_path}")

    augmented_df = pd.DataFrame(new_rows)
    augmented_df.to_csv(AUGMENTED_OUTPUT_CSV, index=False)
    print(f"Saved {len(new_rows)} augmented images and metadata to {AUGMENTED_OUTPUT_CSV}")

    return augmented_df

def show_augmented_example(img, corners):
    augmented = augment(image=img, keypoints=corners.reshape(-1,2))
    aug_img = augmented['image']
    aug_corners = np.array(augmented['keypoints'])

    plt.figure(figsize=(4,4))
    plt.imshow(aug_img.squeeze(), cmap='gray')
    plt.scatter(aug_corners[:,0], aug_corners[:,1], color='red', s=30)
    plt.title("Example Augmentation")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("apriltag_train_data.csv")
    df_ranged = df.iloc[6731:]

    augmented_df = augment_and_save_images(df_ranged)
