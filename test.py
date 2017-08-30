import cv2
import cPickle

from utilities import visualize_gt_label

project_dir = "/home/fregu856/2D_detection/"
# project_dir = "/root/2D_detection"

augmented_train_label_paths = cPickle.load(open(project_dir + "data/augmented_train_label_paths.pkl"))
augmented_train_img_paths = cPickle.load(open(project_dir + "data/augmented_train_img_paths.pkl"))

for img_path, label_path in zip(augmented_train_img_paths, augmented_train_label_paths):
    img_with_bboxes = visualize_gt_label(img_path, label_path)
    cv2.imwrite(img_path.split(".png")[0] + "_gt.png", img_with_bboxes)
