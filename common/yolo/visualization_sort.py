import cv2
import numpy as np
from common.yolo.visualization import bbox_colors


def draw_bboxes(image, np_data: np.array, labels: list = None):
    """
    Draw bounding boxes with labels on the image.

    :param image: cv2 image
    :param np_data: Numpy array [id, lx, ly, rx, ry, conf, cls, ...], size [N, 58] or [N, 7]
    :param labels: list of strings
    :return:
    """

    # Iterate over each bounding box in np_data
    for i in range(np_data.shape[0]):
        # Get bounding box coordinates and convert them to integers
        lx, ly, rx, ry = map(int, np_data[i, 1:5])

        # Get the color for this bounding box based on its id
        box_color = bbox_colors[int(np_data[i, 0]) % len(bbox_colors)]

        # Determine the label text
        if labels is not None:
            label = f"{labels[int(np_data[i, 6])]}: {np_data[i, 5]:.2f}"
        else:
            label = f"ID: {int(np_data[i, 0])}: {np_data[i, 5]:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

        # Get text size for background size
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle as background for text
        cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), box_color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Return the image with bounding boxes drawn
    return image
