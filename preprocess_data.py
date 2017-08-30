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

# split the KITTI training data into train/val:
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
    img = cv2.imread(img_path, -1)

    img_rescaled = cv2.resize(img, (new_img_width, new_img_height))
    img_rescaled_path = project_dir + "data/" + img_id + "_rescaled.png"
    cv2.imwrite(img_rescaled_path, img_rescaled)
    orig_train_img_paths.append(img_rescaled_path)

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

cPickle.dump(train_label_paths,
            open(project_dir + "data/train_label_paths.pkl", "w"))
cPickle.dump(train_img_paths,
            open(project_dir + "data/train_img_paths.pkl", "w"))
# train_label_paths = cPickle.load(open(project_dir + "data/train_label_paths.pkl"))
# train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))

cPickle.dump(val_label_paths,
            open(project_dir + "data/val_label_paths.pkl", "w"))
cPickle.dump(val_img_paths,
            open(project_dir + "data/val_img_paths.pkl", "w"))
# val_label_paths = cPickle.load(open(project_dir + "data/val_label_paths.pkl"))
# val_img_paths = cPickle.load(open(project_dir + "data/val_img_paths.pkl"))

# augment the train set by flipping all train images:
augmented_train_img_paths = []
augmented_train_label_paths = []
for step, (img_path, label_path) in enumerate(train_data):
    print step

    img = cv2.imread(img_path, -1)

    img_flipped = cv2.flip(img, 1)
    img_flipped_path = img_path.split(".png")[0] + "_flipped.png"
    cv2.imwrite(img_flipped_path, img_flipped)
    augmented_train_img_paths.append(img_flipped_path)
    augmented_train_img_paths.append(img_path)

    label_flipped_path = label_path.split(".txt")[0] + "_flipped.txt"
    label_flipped_file = open(label_flipped_path, "a")
    with open(label_path) as label_file:
        for line in label_file:
            splitted_line = line.split(" ")
            x_left = float(splitted_line[4])
            x_right = float(splitted_line[6])

            x_right_flipped = str(new_img_width/2 - (x_left - new_img_width/2))
            x_left_flipped = str(new_img_width/2 - (x_right - new_img_width/2))

            new_line = (splitted_line[0] + " " + splitted_line[1] + " " + splitted_line[2] +
                    " " + splitted_line[3] + " " + x_left_flipped + " " + splitted_line[5] +
                    " " + x_right_flipped + " " + splitted_line[7] + " " + splitted_line[8] +
                    " " + splitted_line[9] + " " + splitted_line[10] + " " + splitted_line[11] +
                    " " + splitted_line[12] + " " + splitted_line[13] + " " + splitted_line[14])

            label_flipped_file.write(new_line)

    label_flipped_file.close()

    augmented_train_label_paths.append(label_flipped_path)
    augmented_train_label_paths.append(label_path)

augmented_train_data = zip(augmented_train_img_paths, augmented_train_label_paths)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)

augmented_train_img_paths, augmented_train_label_paths = zip(*augmented_train_data)

cPickle.dump(augmented_train_label_paths,
            open(project_dir + "data/augmented_train_label_paths.pkl", "w"))
cPickle.dump(augmented_train_img_paths,
            open(project_dir + "data/augmented_train_img_paths.pkl", "w"))
# augmented_train_label_paths = cPickle.load(open(project_dir + "data/augmented_train_label_paths.pkl"))
# augmented_train_img_paths = cPickle.load(open(project_dir + "data/augmented_train_img_paths.pkl"))











# cv2.imshow("img", img)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
