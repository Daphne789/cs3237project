from pupil_apriltags import Detector
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def drawAprilTagBorders(detection_item):
    corners = detection[i].corners
    x_coords = []
    y_coords = []
    for corner in corners:
        x_coords.append(corner[0])
        y_coords.append(corner[1])

    center = detection[i].center
    plt.text(center[0], center[1], str(detection[i].tag_id), color='yellow', fontsize=12, ha='center')
    
    for j in range(4):
        curr_x_coord = [x_coords[j], x_coords[(j + 1) % 4]]
        curr_y_coord = [y_coords[j], y_coords[(j + 1) % 4]]
        plt.plot(curr_x_coord, curr_y_coord, c='red')

at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

image = Image.open("Experiments/apriltag_detection/six_apriltags.png")
imgplot = plt.imshow(image)
img_array = np.asarray(image, dtype=np.uint8)
img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY) #change rgba to black and white channels

print("Img array shape:", img_array.shape)

detection = at_detector.detect(img_array)
print(len(detection), "april tags found")

for i in range(len(detection)):
    april_tag_detected = detection[i]
    print(april_tag_detected)
    corners = detection[i].corners
    x_coords = []
    y_coords = []
    for corner in corners:
        x_coords.append(corner[0])
        y_coords.append(corner[1])

    center = detection[i].center
    plt.text(center[0], center[1], str(detection[i].tag_id), color='yellow', fontsize=12, ha='center')
    
    for j in range(4):
        curr_x_coord = [x_coords[j], x_coords[(j + 1) % 4]]
        curr_y_coord = [y_coords[j], y_coords[(j + 1) % 4]]
        plt.plot(curr_x_coord, curr_y_coord, c='red')

plt.tight_layout()
plt.show()
