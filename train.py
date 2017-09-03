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

from utilities import sparse_to_dense, batch_IOU

project_dir = "/home/fregu856/2D_detection/"
#project_dir = "/root/2D_segmentation/"

data_dir = project_dir + "data/"

model_id = "1" # (change this to not overwrite all log data when you train the model)

model = SqueezeDet_model(model_id)

batch_size = model.batch_size
img_height = model.img_height
img_width = model.img_width
no_of_classes = model.no_of_classes

#train_mean_img = cPickle.load(open("data/mean_img.pkl"))

# load the training data from disk
train_img_paths = cPickle.load(open(data_dir + "train_img_paths.pkl"))
train_bboxes_per_img = cPickle.load(open(data_dir + "train_bboxes_per_img.pkl"))
train_data = zip(train_img_paths, train_bboxes_per_img)

# compute the number of batches needed to iterate through the training data:
no_of_train_imgs = len(train_img_paths)
no_of_batches = int(no_of_train_imgs/batch_size)

def evaluate_on_val(batch_size):
    """
    - DOES:
    """

    # TODO!

    val_loss = 0
    return val_loss

def train_data_iterator(batch_size):
    """
    - DOES:
    """

    # TODO! comment this function more carefully, give more high-level descriptions of what's going on

    batch_mask = 0 # can remove this when the function is done
    batch_gt_deltas = 0
    batch_gt_bboxes = 0
    batch_class_labels = 0

    random.shuffle(train_data)
    train_img_paths, train_bboxes_per_img = zip(*train_data)

    batch_pointer = 0
    for step in range(no_of_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)

        # (list of length batch_size, each element is a list of length
        # no_of_gt_bboxes_in_img containing the class labels (0=car, 1=pedestrian
        # etc.) of the ground truth bboxes in the image)
        class_labels_per_img = []

        # (list of length batch_size, each element is a 2D array of shape
        # [no_of_gt_bboxes_in_img, 4], where each row is [center_x, center_y, w, h]
        # of each ground truth bbox in the image)
        gt_bboxes_per_img  = []

        # (list of length batch_size, each element is a list of length
        # no_of_gt_bboxes_in_img, where each element in turn is a list [delta_x,
        # delta_y, delta_w, delta_h] which describes how to transform the assigned
        # anchor into the ground truth bbox for each ground truth bbox in the image)
        gt_deltas_per_img = []

        # (list of length batch_size, each element is a list of length
        # no_of_gt_bboxes_in_img containing the index of the assigned anchor
        # for each ground truth bbox in the image)
        anchor_indices_per_img  = []
        for i in range(batch_size):
            # read the next img:
            img_path = train_img_paths[batch_pointer + i]
            img = cv2.imread(img_path, -1)
            #img = img - train_mean_img
            batch_imgs[i] = img

            img_bboxes = train_bboxes_per_img[batch_pointer + i]
            # (bbox format: [center_x, center_y, w, h, class_label*], *:string)

            img_class_labels = [model.class_string_to_label[b[4]] for b in img_bboxes]
            class_labels_per_img.append(img_class_labels)

            img_gt_bboxes = np.array([[b[0], b[1], b[2], b[3]] for b in img_bboxes])
            # (bbox format: [center_x, center_y, w, h]. img_gt_bboxes has shape
            # [no_of_gt_bboxes_in_img, 4])
            gt_bboxes_per_img.append(img_gt_bboxes)

            img_gt_deltas = []
            img_anchor_indices = []
            assigned_anchor_indices = []
            for gt_bbox in img_gt_bboxes:
                IOUs = batch_IOU(model.anchor_bboxes, gt_bbox)
                # (IOUs has shape [anchors_per_img, ] and contains the IOU
                # between each anchor bbox and gt_bbox)

                anchor_idx = -1
                sorted_anchor_indices_IOU = np.argsort(IOUs)[::-1] # (-1 gives descending order)
                # (the first element of sorted_anchor_indices_IOU is the index
                # of the anchor with the LARGEST IOU with gt_bbox etc.)
                for idx in sorted_anchor_indices_IOU:
                    if IOUs[idx] <= 0:
                        break
                    if idx not in assigned_anchor_indices:
                        assigned_anchor_indices.append(idx)
                        anchor_idx = idx
                        break
                if anchor_idx == -1: # (if all available IOUs equal 0:)
                    # choose the available anchor which is closest to the ground
                    # truth bbox w.r.t L2 norm:
                    norms = np.sum(np.square(gt_bbox - model.anchor_bboxes), axis=1)
                    sorted_anchor_indices_norm = np.argsort(norms)
                    for idx in sorted_anchor_indices_norm:
                        if idx not in assigned_anchor_indices:
                            assigned_anchor_indices.append(idx)
                            anchor_idx = idx
                            break
                img_anchor_indices.append(anchor_idx)

                assigned_anchor_bbox = model.anchor_bboxes[anchor_idx]
                anchor_cx, anchor_cy, anchor_w, anchor_h = assigned_anchor_bbox

                gt_cx, gt_cy, gt_w, gt_h = gt_bbox

                gt_delta = [0]*4
                gt_delta[0] = (gt_cx - anchor_cx)/anchor_w
                gt_delta[1] = (gt_cy - anchor_cy)/anchor_h
                gt_delta[2] = np.log(gt_w/anchor_w)
                gt_delta[3] = np.log(gt_h/anchor_h)

                img_gt_deltas.append(gt_delta)

            gt_deltas_per_img.append(img_gt_deltas)
            anchor_indices_per_img.append(img_anchor_indices)

        # (we now have batch_imgs, class_labels_per_img, gt_bboxes_per_img,
        # gt_deltas_per_img and anchor_indices_per_img)

        # TODO! comment this final section much better!

        class_label_indices = []
        mask_indices = []
        gt_bbox_indices = []
        gt_delta_values =[]
        gt_bbox_values  = []
        for i in range(batch_size):
            no_of_gt_bboxes_in_img = len(class_labels_per_img[i])

            img_class_labels = class_labels_per_img[i]
            img_anchor_indices = anchor_indices_per_img[i]
            img_gt_deltas = gt_deltas_per_img[i]
            img_gt_bboxes = gt_bboxes_per_img[i]
            for j in range(no_of_gt_bboxes_in_img):
                class_label = img_class_labels[j]
                anchor_idx = img_anchor_indices[j]
                gt_delta = img_gt_deltas[j]
                gt_bbox = img_gt_bboxes[j]

                class_label_indices.append([i, anchor_idx, class_label])
                mask_indices.append([i, anchor_idx])
                gt_bbox_indices.extend([[i, anchor_idx, k] for k in range(4)])
                gt_delta_values.extend(gt_delta)
                gt_bbox_values.extend(gt_bbox)

        # (we now have mask_indices, class_label_indices, gt_bbox_indices,
        # gt_delta_values and gt_bbox_values)

        batch_mask = sparse_to_dense(mask_indices, [batch_size, model.anchors_per_img],
                    [1.0]*len(mask_indices))
        batch_mask = np.reshape(batch_mask, [batch_size, model.anchors_per_img, 1])

        batch_gt_deltas = sparse_to_dense(gt_bbox_indices,
                    [batch_size, model.anchors_per_img, 4], gt_delta_values)

        batch_gt_bboxes = sparse_to_dense(gt_bbox_indices,
                    [batch_size, model.anchors_per_img, 4], gt_bbox_values)

        batch_class_labels = sparse_to_dense(class_label_indices,
                    [batch_size, model.anchors_per_img, no_of_classes],
                    [1.0]*len(class_label_indices))

        batch_pointer += batch_size

        yield (batch_imgs, batch_mask, batch_gt_deltas, batch_gt_bboxes, batch_class_labels)

no_of_epochs = 100

# create a saver for saving all model variables/parameters:
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

# initialize all log data containers:
train_loss_per_epoch = []
val_loss_per_epoch = []

# initialize a list containing the 5 best val losses (is used to tell when it
# makes sense to save a model checkpoint):
best_epoch_losses = [1000, 1000, 1000, 1000, 1000]

with tf.Session() as sess:
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(no_of_epochs):
        print "###########################"
        print "######## NEW EPOCH ########"
        print "###########################"
        print "epoch: %d/%d" % (epoch+1, no_of_epochs)

        # run an epoch and get all batch losses:
        batch_losses = []
        for step, (imgs, mask, gt_deltas, gt_bboxes, class_labels) in enumerate(train_data_iterator()):
            # create a feed dict containing the batch data:
            batch_feed_dict = model.create_feed_dict(imgs, 0.8, mask=mask,
                        gt_deltas=gt_deltas, gt_bboxes=gt_bboxes,
                        class_labels=class_labels)

            # compute the batch loss and compute & apply all gradients w.r.t to
            # the batch loss (without model.train_op in the call, the network
            # would NOT train, we would only compute the batch loss):
            batch_loss, _ = sess.run([model.loss, model.train_op],
                        feed_dict=batch_feed_dict)
            batch_losses.append(batch_loss)

            print "step: %d/%d, training batch loss: %g" % (step+1, no_of_batches, batch_loss)

        # compute the train epoch loss:
        train_epoch_loss = np.mean(batch_losses)
        # save the train epoch loss:
        train_loss_per_epoch.append(train_epoch_loss)
        # save the train epoch losses to disk:
        cPickle.dump(train_loss_per_epoch, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "w"))
        print "training loss: %g" % train_epoch_loss

        # run the model on the validation data:
        val_loss = evaluate_on_val(batch_size, sess)
        # save the val epoch loss:
        val_loss_per_epoch.append(val_loss)
        # save the val epoch losses to disk:
        cPickle.dump(val_loss_per_epoch, open("%sval_loss_per_epoch.pkl"\
                    % model.model_dir, "w"))
        print "validaion loss: %g" % val_loss

        if val_loss < max(best_epoch_losses): # (if top 5 performance on val:)
            # save the model weights to disk:
            checkpoint_path = (model.checkpoints_dir + "model_" +
                        model.model_id + "_epoch_" + str(epoch + 1) + ".ckpt")
            saver.save(sess, checkpoint_path)
            print "checkpoint saved in file: %s" % checkpoint_path

            # update the top 5 val losses:
            index = best_epoch_losses.index(max(best_epoch_losses))
            best_epoch_losses[index] = val_loss

        # plot the training loss vs epoch and save to disk:
        plt.figure(1)
        plt.plot(train_loss_per_epoch, "k^")
        plt.plot(train_loss_per_epoch, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("training loss per epoch")
        plt.savefig("%strain_loss_per_epoch.png" % model.model_dir)
        plt.close(1)

        # plot the val loss vs epoch and save to disk:
        plt.figure(1)
        plt.plot(val_loss_per_epoch, "k^")
        plt.plot(val_loss_per_epoch, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("validation loss per epoch")
        plt.savefig("%sval_loss_per_epoch.png" % model.model_dir)
        plt.close(1)

        # TODO! get more info during training!
        # op_list = [model.train_op, model.loss, model.det_boxes,
        #            model.det_probs, model.det_class, model.conf_loss,
        #            model.bbox_loss, model.class_loss]
        # _, loss_value, det_boxes, det_probs, det_class conf_loss, bbox_loss, class_loss = sess.run(op_list, feed_dict=feed_dict)
