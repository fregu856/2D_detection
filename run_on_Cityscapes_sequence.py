import numpy as np
import cPickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import cv2
import random
import os

from model import SqueezeDet_model

from utilities import draw_bboxes

#project_dir = "/home/fregu856/2D_detection/"
project_dir = "/root/2D_detection/"

data_dir = project_dir + "data/"

model_id = "Cityscapes_seq_run"

model = SqueezeDet_model(model_id)

batch_size = model.batch_size
img_height = model.img_height
img_width = model.img_width
no_of_classes = model.no_of_classes

train_mean_channels = cPickle.load(open("data/mean_channels.pkl"))

# load the test data from disk
frame_paths = cPickle.load(open(data_dir + "Cityscapes_seq_0_frame_paths.pkl"))

# compute the number of batches needed to iterate through the data:
no_of_frames = len(frame_paths)
no_of_batches = int(no_of_frames/batch_size)

results_dir = model.project_dir + "results_on_Cityscapes_seq/"

# create a saver for saving all model variables/parameters:
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    #saver.restore(sess, "/home/fregu856/2D_detection/training_logs/best_model/model_1_epoch_58.ckpt")
    saver.restore(sess, "/root/2D_detection/training_logs/best_model/model_1_epoch_58.ckpt")

    batch_pointer = 0
    for step in range(no_of_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)

        img_paths = []
        for i in range(batch_size):
            # read the next img:
            img_path = frame_paths[batch_pointer + i]
            img_paths.append(img_path)
            img = cv2.imread(img_path, -1)

            img = cv2.resize(img, (img_width, int(img_height*(float(img_width)/float(2048)))))
            img = img[img_height:]

            img = cv2.resize(img, (img_width, img_height))
            img = img - train_mean_channels
            batch_imgs[i] = img
        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(batch_imgs, 1.0)

        pred_bboxes, detection_classes, detection_probs  = sess.run([model.pred_bboxes,
                    model.detection_classes, model.detection_probs],
                    feed_dict=batch_feed_dict)
        print "step: %d/%d" % (step + 1, no_of_batches)

        for i in range(batch_size):
            final_bboxes, final_probs, final_classes = model.filter_prediction(pred_bboxes[i],
                        detection_probs[i], detection_classes[i])

            keep_idx = [idx for idx in range(len(final_probs)) if final_probs[idx] > model.plot_prob_thresh]
            final_bboxes = [final_bboxes[idx] for idx in keep_idx]
            final_probs = [final_probs[idx] for idx in keep_idx]
            final_classes = [final_classes[idx] for idx in keep_idx]

            # draw the bboxes outputed by the model:
            pred_img = draw_bboxes(batch_imgs[i].copy()+train_mean_channels, final_bboxes, final_classes, final_probs)
            img_name = img_paths[i].split("/")[-1]
            pred_path = results_dir + img_name.split(".png")[0] + "_pred.png"
            cv2.imwrite(pred_path, pred_img)

fourcc = cv2.cv.CV_FOURCC("M", "J", "P", "G")
out = cv2.VideoWriter(results_dir + "Cityscapes_seq_0_pred.avi", fourcc, 10.0, (img_width, img_height))

frame_names = sorted(os.listdir(results_dir))
for step, frame_name in enumerate(frame_names):
    if step % 100 == 0:
        print step

    if ".png" in frame_name:
        frame_path = results_dir + frame_name
        frame = cv2.imread(frame_path, -1)

        out.write(frame)
