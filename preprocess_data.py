import cv2
import cPickle
import os
import numpy as np
import random

from utilities import bbox_transform_inv

#project_dir = "/home/fregu856/2D_detection/"
#data_dir = "/home/fregu856/data/"
project_dir = "/root/2D_detection/"
data_dir = "/root/data/"

KITTI_dir = data_dir + "KITTI/"

new_img_height = 375 # (the height all images fed to the model will be resized to)
new_img_width = 1242 # (the width all images fed to the model will be resized to)
no_of_classes = 3 # (number of object classes: cars, pedestrians, bicyclists)

# split the KITTI train data into train/val:
orig_train_imgs_dir = KITTI_dir + "/data_object/training/image_2/"
orig_train_labels_dir = KITTI_dir + "/data_object/training/label_2/"

# # get the path to all KITTI train imgs and their corresponding label file:
orig_train_img_paths = []
orig_train_label_paths = []
orig_train_img_names = os.listdir(orig_train_imgs_dir)
for step, img_name in enumerate(orig_train_img_names):
    if step % 100 == 0:
        print step

    img_id = img_name.split(".png")[0]

    label_path = orig_train_labels_dir + img_id + ".txt"
    orig_train_label_paths.append(label_path)

    img_path = orig_train_imgs_dir + img_name
    orig_train_img_paths.append(img_path)

# # randomly shuffle the KITTI train data:
orig_train_data = zip(orig_train_img_paths, orig_train_label_paths)
random.shuffle(orig_train_data)
random.shuffle(orig_train_data)
random.shuffle(orig_train_data)
random.shuffle(orig_train_data)

# # select 80 % of the imgs as train data, 20 % as val:
no_of_imgs = len(orig_train_img_paths)
train_data = orig_train_data[:int(no_of_imgs*0.8)]
val_data = orig_train_data[-int(no_of_imgs*0.2):]
print "number of val imgs: %d" % len(val_data)
print "number of train imgs before augmentation: %d " % len(train_data)

# # save the val data to disk for later use:
val_img_paths, val_label_paths = zip(*val_data)
cPickle.dump(val_label_paths,
            open(project_dir + "data/val_label_paths.pkl", "w"))
cPickle.dump(val_img_paths,
            open(project_dir + "data/val_img_paths.pkl", "w"))
# val_label_paths = cPickle.load(open(project_dir + "data/val_label_paths.pkl"))
# val_img_paths = cPickle.load(open(project_dir + "data/val_img_paths.pkl"))


# augment the train data by flipping all train imgs:
augmented_train_img_paths = []
augmented_train_label_paths = []
for step, (img_path, label_path) in enumerate(train_data):
    if step % 100 == 0:
        print step

    img = cv2.imread(img_path, -1)

    # flip the img and save to project_dir/data:
    img_flipped = cv2.flip(img, 1)
    img_flipped_path = img_path.split(".png")[0] + "_flipped.png"
    img_flipped_path = project_dir + "data/" + img_flipped_path.split("/image_2/")[1]
    cv2.imwrite(img_flipped_path, img_flipped)
    # save the paths to the flipped and original imgs (NOTE! the order must match
    # the order we apend the label paths below):
    augmented_train_img_paths.append(img_flipped_path)
    augmented_train_img_paths.append(img_path)

    # modify the corresponding label file to match the flipping and save to
    # project_dir/data:
    label_flipped_path = label_path.split(".txt")[0] + "_flipped.txt"
    label_flipped_path = project_dir + "data/" + label_flipped_path.split("/label_2/")[1]
    label_flipped_file = open(label_flipped_path, "w")
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

    # save the paths to the flipped and original label files (NOTE! the order must
    # match the order we append the img paths above):
    augmented_train_label_paths.append(label_flipped_path)
    augmented_train_label_paths.append(label_path)

# # randomly shuffle the augmented train data:
augmented_train_data = zip(augmented_train_img_paths, augmented_train_label_paths)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)
random.shuffle(augmented_train_data)

# # save the augmented train data to disk for later use:
train_data = augmented_train_data
train_img_paths, train_label_paths = zip(*train_data)
no_of_train_imgs = len(train_img_paths)
cPickle.dump(train_label_paths,
            open(project_dir + "data/train_label_paths.pkl", "w"))
cPickle.dump(train_img_paths,
            open(project_dir + "data/train_img_paths.pkl", "w"))
# train_label_paths = cPickle.load(open(project_dir + "data/train_label_paths.pkl"))
# train_img_paths = cPickle.load(open(project_dir + "data/train_img_paths.pkl"))
print "number of train imgs after augmentation: %d " % len(train_data)


# compute the mean pixel channels of the train imgs:
no_of_train_imgs = len(train_img_paths)
mean_channels = np.zeros((3, ))
for step, img_path in enumerate(train_img_paths):
    if step % 100 == 0:
        print step

    img = cv2.imread(img_path, -1)

    img_mean_channels = np.mean(img, axis=0)
    img_mean_channels = np.mean(img_mean_channels, axis=0)

    mean_channels += img_mean_channels

mean_channels = mean_channels/float(no_of_train_imgs)

# # save to disk for later use:
cPickle.dump(mean_channels, open(project_dir + "data/mean_channels.pkl", "w"))


# read all relevant bboxes (bounding boxes) from the train labels:
# # (train_bboxes_per_img is a list of length no_of_train_imgs where each element
# # is a list containing that img's bboxes)
train_bboxes_per_img = []
for step, label_path in enumerate(train_label_paths):
    if step % 100 == 0:
        print step

    bboxes = []
    with open(label_path) as label_file:
        for line in label_file:
            splitted_line = line.split(" ")
            bbox_class = splitted_line[0].lower().strip()
            if bbox_class not in ["car", "cyclist", "pedestrian"]:
                break
            x_min = float(splitted_line[4])
            y_min = float(splitted_line[5])
            x_max = float(splitted_line[6])
            y_max = float(splitted_line[7])

            c_x, c_y, w, h = bbox_transform_inv([x_min, y_min, x_max, y_max])
            bboxes.append([c_x, c_y, w, h, bbox_class])

    train_bboxes_per_img.append(bboxes)

# # save to disk for later use:
cPickle.dump(train_bboxes_per_img,
            open(project_dir + "data/train_bboxes_per_img.pkl", "w"))


# read all relevant bboxes from the val labels:
val_bboxes_per_img = []
for step, label_path in enumerate(val_label_paths):
    if step % 100 == 0:
        print step

    bboxes = []
    with open(label_path) as label_file:
        for line in label_file:
            splitted_line = line.split(" ")
            bbox_class = splitted_line[0].lower().strip()
            if bbox_class not in ["car", "cyclist", "pedestrian"]:
                break
            x_min = float(splitted_line[4])
            y_min = float(splitted_line[5])
            x_max = float(splitted_line[6])
            y_max = float(splitted_line[7])

            c_x, c_y, w, h = bbox_transform_inv([x_min, y_min, x_max, y_max])
            bboxes.append([c_x, c_y, w, h, bbox_class])

    val_bboxes_per_img.append(bboxes)

# # save to disk for later use:
cPickle.dump(val_bboxes_per_img,
            open(project_dir + "data/val_bboxes_per_img.pkl", "w"))


# read, resize and save frames from a private dash cam video (to qualitatively
# test the model output after training):
cap = cv2.VideoCapture(data_dir + "trollhattan_video/trollhattan_video.mp4")
trollhattan_frame_paths = []
counter = 0
while True:
    # capture frame-by-frame:
    ret, frame = cap.read()
    if counter % 3 == 0 and ((counter > 34600 and counter < 37030) or (counter > 27500 and counter < 29370)):
        print counter

        # resize by cropping the bottom left part of the image of size
        # (new_img_height, new_img_width):
        frame = frame[new_img_height:, :new_img_width]

        frame_path = data_dir + "trollhattan_video/" + str(counter) + ".png"
        trollhattan_frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)

    counter += 1
    if counter > 40000:
        break

cPickle.dump(trollhattan_frame_paths,
            open(project_dir + "data/trollhattan_frame_paths.pkl", "w"))


# get and save the paths to all frames in the KITTI test sequence 0000 (to
# qualitatively test the model output after training):
KITTI_seq_test_0_frame_paths = []
KITTI_seq_dir = KITTI_dir + "data_tracking/testing/image_02/0000/"
frame_names = os.listdir(KITTI_seq_dir)
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    frame_path = KITTI_dir + "data_tracking/testing/image_02/0000/" + frame_name
    KITTI_seq_test_0_frame_paths.append(frame_path)
cPickle.dump(KITTI_seq_test_0_frame_paths,
            open(project_dir + "data/KITTI_seq_test_0_frame_paths.pkl", "w"))

# get and save the paths to all frames in the KITTI test sequence 0001 (to
# qualitatively test the model output after training):
KITTI_seq_test_1_frame_paths = []
KITTI_seq_dir = KITTI_dir + "data_tracking/testing/image_02/0001/"
frame_names = os.listdir(KITTI_seq_dir)
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    frame_path = KITTI_dir + "data_tracking/testing/image_02/0001/" + frame_name
    KITTI_seq_test_1_frame_paths.append(frame_path)
cPickle.dump(KITTI_seq_test_1_frame_paths,
            open(project_dir + "data/KITTI_seq_test_1_frame_paths.pkl", "w"))

# get and save the paths to all frames in the KITTI test sequence 0004 (to
# qualitatively test the model output after training):
KITTI_seq_test_4_frame_paths = []
KITTI_seq_dir = KITTI_dir + "data_tracking/testing/image_02/0004/"
frame_names = os.listdir(KITTI_seq_dir)
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    frame_path = KITTI_dir + "data_tracking/testing/image_02/0004/" + frame_name
    KITTI_seq_test_4_frame_paths.append(frame_path)
cPickle.dump(KITTI_seq_test_4_frame_paths,
            open(project_dir + "data/KITTI_seq_test_4_frame_paths.pkl", "w"))

# get and save the paths to all frames in the KITTI test sequence 0012 (to
# qualitatively test the model output after training):
KITTI_seq_test_12_frame_paths = []
KITTI_seq_dir = KITTI_dir + "data_tracking/testing/image_02/0012/"
frame_names = os.listdir(KITTI_seq_dir)
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    frame_path = KITTI_dir + "data_tracking/testing/image_02/0012/" + frame_name
    KITTI_seq_test_12_frame_paths.append(frame_path)
cPickle.dump(KITTI_seq_test_12_frame_paths,
            open(project_dir + "data/KITTI_seq_test_12_frame_paths.pkl", "w"))
