from common.yolo.yolo_results import Yolo, YoloPose, YoloPoint
import numpy as np


###### Convert yolo.results to a list of Yolo/YoloPose objects ######
def results_to_yolo_list(results) -> list[Yolo]:
    # A list to store the results
    yolo_objects = []

    # If the results is not a list
    if isinstance(results, list):
        raise ValueError("The results object is a list.")

    # If boxes is not in the results
    if not hasattr(results, 'boxes'):
        raise ValueError("The results object does not have the attribute 'boxes'.")

    # Loop through the results
    for det in results.boxes:
        # Get the bounding box
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        # Get the confidence
        # conf = det.conf # CUDA tensor
        conf = det.conf.cpu().numpy().item()

        # Get the class
        cls = int(det.cls[0])

        # Create a yolo object to store the results
        yolo = Yolo(x1, y1, x2, y2, cls, conf)

        # Append the yolo object to the list
        yolo_objects.append(yolo)

    return yolo_objects


def results_to_pose_list(results) -> list[YoloPose]:
    # A list to store the results
    yolo_objects = []

    # If the results is a list
    if isinstance(results, list):
        raise ValueError("The results object is a list.")

    # If boxes is not in the results
    if not hasattr(results, 'boxes'):
        raise ValueError("The results object does not have the attribute 'boxes'.")

    # If keypoints is not in the results
    if not hasattr(results, 'keypoints'):
        raise ValueError("The results object does not have the attribute 'keypoints'.")

    # Loop through the results
    for box, kpts in zip(results.boxes, results.keypoints):
        # Get the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the confidence
        # conf = det.conf # CUDA tensor
        conf = box.conf.cpu().numpy().item()

        # Create a list to store the keypoints
        keypoints = []

        # Squeeze the keypoints
        kpts_xy = kpts.xy.cpu().numpy().squeeze()  # (17, 2)
        kpts_conf = kpts.conf.cpu().numpy().squeeze()  # (17,)

        # Loop through the keypoints
        for (x, y), c in zip(kpts_xy, kpts_conf):
            # Create a yolo point object
            point = YoloPoint(int(x), int(y), c)

            # Append the yolo point object to the list
            keypoints.append(point)

        # Create a yolo pose object to store the results
        pose = YoloPose(x1, y1, x2, y2, conf, keypoints)

        # Append the yolo pose object to the list
        yolo_objects.append(pose)

    return yolo_objects


###### Convert a list of Yolo/YoloPose objects to a numpy array ######
def yolo_list_to_numpy(data: list[Yolo]) -> np.ndarray:
    """
    Converts a list of Yolo objects to a numpy array.

    :param data: list of Yolo objects
    :return: numpy array
    """
    return np.array([y.to_list() for y in data])


def pose_list_to_numpy(data: list[YoloPose]) -> np.ndarray:
    """
    Converts a list of YoloPose objects to a numpy array. The main attributes are
    converted into one part of the array, while the 'pts' attribute is left as a
    list of arrays.
    """
    # Convert the main attributes to a numpy array and keep pts as an object
    return np.array([[y.lx, y.ly, y.rx, y.ry, y.conf, [pt.to_list() for pt in y.pts]] for y in data], dtype=object)


###### Convert a numpy array to a list of Yolo/YoloPose objects ######
def numpy_to_yolo_list(data: np.ndarray) -> list[Yolo]:
    """
    Converts a numpy array to a list of Yolo objects.

    :param data: numpy array
    :return: list of Yolo objects
    """
    output = []
    for row in data:
        y = Yolo()
        y.from_list(row.tolist())
        output.append(y)

    return output

def numpy_to_pose_list(data: np.ndarray) -> list[YoloPose]:
    """
    Converts a numpy array to a list of YoloPose objects.

    :param data: numpy array
    :return: list of YoloPose objects
    """
    output = []
    for row in data:
        y = YoloPose()
        y.from_list(row.tolist())
        output.append(y)

    return output