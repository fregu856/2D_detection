import numpy as np
import cPickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import cv2
import random

from model import SqueezeDet_model

from utilities import draw_bboxes

#project_dir = "/home/fregu856/2D_detection/"
project_dir = "/root/2D_detection/"

data_dir = project_dir + "data/"

model_id = "test_eval"

model = SqueezeDet_model(model_id)

batch_size = model.batch_size
img_height = model.img_height
img_width = model.img_width
no_of_classes = model.no_of_classes

train_mean_channels = cPickle.load(open("data/mean_channels.pkl"))

# load the test data from disk
test_img_paths = cPickle.load(open(data_dir + "test_img_paths.pkl"))

# compute the number of batches needed to iterate through the test data:
no_of_test_imgs = len(test_img_paths)
no_of_test_batches = int(no_of_test_imgs/batch_size)

# create a saver for saving all model variables/parameters:
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    #saver.restore(sess, "/home/fregu856/2D_detection/training_logs/best_model/model_1_epoch_58.ckpt")
    saver.restore(sess, "/root/2D_detection/training_logs/best_model/model_1_epoch_58.ckpt")

    batch_pointer = 0
    for step in range(no_of_test_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)

        img_paths = []
        for i in range(batch_size):
            # read the next img:
            img_path = test_img_paths[batch_pointer + i]
            img_paths.append(img_path)
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (img_width, img_height))
            img = img - train_mean_channels
            batch_imgs[i] = img
        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(batch_imgs, 1.0)

        pred_bboxes, detection_classes, detection_probs  = sess.run([model.pred_bboxes,
                    model.detection_classes, model.detection_probs],
                    feed_dict=batch_feed_dict)
        print "test step: %d/%d" % (step+1, no_of_test_batches)

        for i in range(batch_size):
            final_bboxes, final_probs, final_classes = model.filter_prediction(pred_bboxes[i],
                        detection_probs[i], detection_classes[i])

            keep_idx = [idx for idx in range(len(final_probs)) if final_probs[idx] > model.plot_prob_thresh]
            final_bboxes = [final_bboxes[idx] for idx in keep_idx]
            final_probs = [final_probs[idx] for idx in keep_idx]
            final_classes = [final_classes[idx] for idx in keep_idx]

            # draw the bboxes outputed by the model:
            pred_img = draw_bboxes(batch_imgs[i].copy()+train_mean_channels, final_bboxes, final_classes, final_probs)
            img_name = img_paths[i].split("image_2/")[1]
            pred_path = model.project_dir + "results_on_test/" + img_name.split(".png")[0] + "_pred.png"
            cv2.imwrite(pred_path, pred_img)
