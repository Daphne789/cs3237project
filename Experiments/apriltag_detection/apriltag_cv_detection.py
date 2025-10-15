import cv2
from .basic_apriltag_roi_detection import *

def detect_apriltag_from_cv(cap, detector):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_apriltag_from_array(gray, detector,is_plot=False)
        
        for det in detections:
            corners = det.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            centre = tuple(det.center.astype(int))
            cv2.putText(frame, str(det.tag_id), centre, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("April tag detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
