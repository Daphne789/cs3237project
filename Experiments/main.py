from apriltag_detection.apriltag_cv_detection import *
from apriltag_detection.AprilTag import AprilTag
import os
import pandas as pd

if __name__ == "__main__b":
    detector = initialise_detector()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        detect_apriltag_from_cv(cap, detector)
    
if __name__  == "__main1__":
    detector = initialise_detector()
    img_filepath = input("Input path: ")
    # image_folder_path = "captured_frames"
    # img_filepath = os.path.join(image_folder_path, "frame_52.jpg")
    detection = detect_apriltag_from_image(img_filepath, detector)
    has_apriltag = len(detection) != 0
    
    print("Image filepath:", img_filepath)
    print("has apriltag:", has_apriltag)
    corners = detection[0].corners.flatten()
    print("corners:", corners.flatten())
    
# Converts to CSV file
if __name__ == "__main__":
    detector = initialise_detector()
    image_folder_path = "captured_frames"
    data = generate_training_data(detector, image_folder_path)

if __name__ == "__main__":
    # print percentage of images with detectable apriltags
    data = pd.read_csv("apriltag_train_data.csv")
    apriltag_count = len(data[data['has_apriltag']])
    total_rows = len(data)
    print("April tags detected:", apriltag_count)
    print(f"April tag percentage: {(100 * apriltag_count / total_rows):.2f}%")
