import cv2
import tensorflow as tf

def visualize_gt_label(img_path, label_path):
    class_to_color = {"Car": (255, 191, 0),
                      "Cyclist": (0, 191, 255),
                      "Pedestrian": (255, 0, 191)}

    img = cv2.imread(img_path, -1)

    with open(label_path) as label_file:
        for line in label_file:
            splitted_line = line.split(" ")
            bbox_class = splitted_line[0]
            if bbox_class not in ["Car", "Cyclist", "Pedestrian"]:
                break
            x_left = int(float(splitted_line[4]))
            y_top = int(float(splitted_line[5]))
            x_right = int(float(splitted_line[6]))
            y_bottom = int(float(splitted_line[7]))

            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), class_to_color[bbox_class], 2)

    img_with_bboxes = img
    return img_with_bboxes

# (taken from the official implementation:)
def safe_exp(w, thresh):
    """
    safe exponential function for tensors
    """

    slope = np.exp(thresh)

    lin_bool = w > thresh
    lin_region = tf.to_float(lin_bool)

    lin_out = slope*(w - thresh + 1.)
    exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out

    return out

# (modified from the official implementation:)
def bbox_transform(bbox):
    """
    convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    """

    cx, cy, w, h = bbox

    xmin = cx - w/2
    ymin = cy - h/2
    xmax = cx + w/2
    ymax = cy + h/2

    out_box = [xmin, ymin, xmax, ymax]

    return out_box

# (modified from the official implementation:)
def bbox_transform_inv(bbox):
    """
    convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
    """

    xmin, ymin, xmax, ymax = bbox

    w = xmax - xmin + 1.0
    h = ymax - ymin + 1.0
    cx  = xmin + width/2
    cy  = ymin + height/2

    out_box = [cx, cy, w, h]

    return out_box
