import cv2
import cPickle
import os
import numpy as np
import random

project_dir = "/home/fregu856/2D_detection/"
KITTI_dir = "/home/fregu856/data/KITTI/"

# project_dir = "/root/2D_detection"
# cityscapes_dir = "/root/KITTI/"

new_img_height = 375
new_img_width = 1242
no_of_classes = 3

orig_train_imgs_dir = KITTI_dir + "/data_object/training/image_2/"
orig_train_labels_dir = KITTI_dir + "/data_object/training/label_2/"

orig_train_img_paths = []
orig_train_label_paths = []

orig_train_img_names = os.listdir(orig_train_imgs_dir)
for step, img_name in enumerate(orig_train_img_names):
    print step

    img_id = img_name.split(".png")[0]
    label_path = orig_train_labels_dir + img_id + ".txt"
    orig_train_label_paths.append(label_path)

    img_path = orig_train_imgs_dir + img_name
    orig_train_img_paths.append(img_path)

orig_train_data = zip(orig_train_img_paths, orig_train_label_paths)
random.shuffle(orig_train_data)
random.shuffle(orig_train_data)
random.shuffle(orig_train_data)
random.shuffle(orig_train_data)

no_of_imgs = len(orig_train_img_paths)
train_data = orig_train_data[:int(no_of_imgs*0.8)]
val_data = orig_train_data[-int(no_of_imgs*0.2):]

train_img_paths, train_label_paths = zip(*train_data)
val_img_paths, val_label_paths = zip(*val_data)






    # img = cv2.imread(img_path, -1)
    #
    # # cv2.imshow("img", img)
    # # if cv2.waitKey(1) & 0xFF == ord('q'):
    # #     break
    #
    # # img_small = cv2.resize(img, (new_img_width, new_img_height), interpolation=cv2.INTER_NEAREST)
    # # img_small_path = project_dir + "data/" + img_id + ".png"
    # # cv2.imwrite(img_small_path, img_small)
    # # train_img_paths.append(img_small_path)
