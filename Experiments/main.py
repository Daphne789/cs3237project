from apriltag_detection.apriltag_cv_detection import *
import os

if __name__ == "__main__b":
    detector = initialise_detector()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        detect_apriltag_from_cv(cap, detector)
    
if __name__  == "__main__1":
    detector = initialise_detector()
    image_folder_path = "captured_frames"
    img_filepath = os.path.join(image_folder_path, "frame_445.jpg")
    detection = detect_apriltag_from_image(img_filepath, detector, is_plot=False)
    has_apriltag = len(detection) != 0
    
    print("Image filepath:", img_filepath)
    print("has apriltag:", has_apriltag)
    corners = detection[0].corners.flatten()
    print("corners:", corners.flatten())
    
if __name__ == "__main__":
    detector = initialise_detector()
    image_folder_path = "captured_frames"
    data = generate_training_data(detector, image_folder_path)