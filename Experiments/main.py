from apriltag_detection.apriltag_cv_detection import *

if __name__ == "__main__":
    at_detector = initialise_detector()
    cap = cv2.VideoCapture(0)
    detect_apriltag_from_cv(cap, at_detector)
    