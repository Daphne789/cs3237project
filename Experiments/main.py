from apriltag_detection.basic_apriltag_roi_detection import detect_apriltag_from_image

if __name__ == "__main__":
    filepath = "Experiments/apriltag_detection/mirror_apriltags.png"
    detect_apriltag_from_image(filepath)
