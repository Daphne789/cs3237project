import numpy as np
import cv2


# Example inputs (replace with real values)
# [[111.62237549 112.32441711]
#  [194.61705017 121.10914612]
#  [203.26782227  37.59911728]
#  [120.57845306  32.17260361]]
def calc_dist(corners):
    image_points = np.array(
        [
            [corners[3]],  # pixel coords of top-left corner
            [corners[2]],  # top-right
            [corners[1]],  # bottom-right
            [corners[0]],  # bottom-left
        ],
        dtype=np.float32,
    )

    S = 0.10  # tag side in meters

    object_points = np.array(
        [
            [-S / 2, S / 2, 0.0],
            [S / 2, S / 2, 0.0],
            [S / 2, -S / 2, 0.0],
            [-S / 2, -S / 2, 0.0],
        ],
        dtype=np.float32,
    )

    # Camera intrinsics
    K = np.array([[320, 0, 160], [0, 320, 120], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed")

    # tvec is (3x1) in meters (object -> camera)
    tvec = tvec.reshape(3)
    forward_distance = tvec[2]  # distance along camera Z-axis
    euclidean_distance = np.linalg.norm(tvec)  # full 3D distance

    # print("tvec (m):", tvec)
    # print("forward distance (m):", forward_distance)
    # print("euclidean distance (m):", euclidean_distance)

    return forward_distance