import cv2

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
