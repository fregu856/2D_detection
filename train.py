import numpy as np
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import cv2

from model import SqueezeDet_model
from utilities import sparse_to_dense

project_dir = "/home/fregu856/2D_detection/"
#project_dir = "/root/2D_segmentation/"

data_dir = project_dir + "data/"

img_height = 375
img_width = 1242
no_of_classes = 3

train_mean_img = cPickle.load(open("data/mean_img.pkl"))

def evaluate_on_val(batch_size, sess):
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

    # TODO!

    batch_mask = 0 # can remove this when the function is done
    batch_bbox_deltas = 0
    batch_bboxes = 0
    batch_labels = 0

    # load the training data from disk
    train_img_paths = cPickle.load(open(data_dir + "train_img_paths.pkl"))
    train_bboxes_per_img = cPickle.load(open(data_dir + "train_bboxes_per_img.pkl"))

    # TODO! should I perhaps always shuffle the data here?

    # compute the number of batches needed to iterate through the training data:
    global no_of_batches
    no_of_train_imgs = len(train_img_paths)
    no_of_batches = int(no_of_train_imgs/batch_size)

    batch_pointer = 0
    for step in range(no_of_batches):
        # get and yield the next batch_size imgs and ******* from the training data:
        batch_imgs = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(train_img_paths[batch_pointer + i], -1)
            img = img - train_mean_img
            batch_imgs[i] = img




            bboxes = train_bboxes_per_img[batch_pointer + i]
            # TODO! transform this to everything we need to yield




        batch_pointer += batch_size

        yield (batch_imgs, batch_mask, batch_bbox_deltas, batch_bboxes, batch_labels)

no_of_epochs = 100
model_id = "1" # (change this to not overwrite all log data when you train the model)

model = SqueezeDet_model(model_id)

batch_size = model.batch_size

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
        for step, (imgs, mask, bbox_deltas, bboxes, labels) in enumerate(train_data_iterator(batch_size)):
            # create a feed dict containing the batch data:
            batch_feed_dict = model.create_feed_dict(imgs, 0.8, input_mask=mask,
                        bbox_delta_input=bbox_deltas, bbox_input=bboxes, labels=labels)

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









def train():
    mc = kitti_squeezeDet_config()
    mc.IS_TRAINING = True
    mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
    model = SqueezeDet(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    def load_data():
        # read batch input
        image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, bbox_per_batch = imdb.read_batch()

        label_indices, bbox_indices, box_delta_values, mask_indices, box_values  = [], [], [], [], []
        aidx_set = set()
        num_discarded_labels = 0
        num_labels = 0
        for i in range(len(label_per_batch)): # batch_size
            for j in range(len(label_per_batch[i])): # number of annotations
                num_labels += 1
                if (i, aidx_per_batch[i][j]) not in aidx_set:
                    aidx_set.add((i, aidx_per_batch[i][j]))
                    label_indices.append(
                        [i, aidx_per_batch[i][j], label_per_batch[i][j]])
                    mask_indices.append([i, aidx_per_batch[i][j]])
                    bbox_indices.extend(
                        [[i, aidx_per_batch[i][j], k] for k in range(4)])
                    box_delta_values.extend(box_delta_per_batch[i][j])
                    box_values.extend(bbox_per_batch[i][j])
                else:
                    num_discarded_labels += 1

        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        box_input = model.ph_box_input
        labels = model.ph_labels

        feed_dict = {
          image_input: image_per_batch,
          input_mask: np.reshape(sparse_to_dense(mask_indices, [mc.BATCH_SIZE, mc.ANCHORS], [1.0]*len(mask_indices)), [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          box_delta_input: sparse_to_dense(bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4], box_delta_values),
          box_input: sparse_to_dense(bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4], box_values),
          labels: sparse_to_dense(label_indices, [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES], [1.0]*len(label_indices))
        }

        return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()
    sess.run(init)

    # try:
    for step in xrange(FLAGS.max_steps):
        feed_dict, image_per_batch, label_per_batch, bbox_per_batch = load_data()
        op_list = [model.train_op, model.loss, model.det_boxes,
                   model.det_probs, model.det_class, model.conf_loss,
                   model.bbox_loss, model.class_loss]
        _, loss_value, det_boxes, det_probs, det_class conf_loss, bbox_loss, class_loss = sess.run(op_list, feed_dict=feed_dict)


def read_batch(model):
    """
    read a batch of image and bounding box annotations

    args:
        shuffle: whether or not to shuffle the dataset

    returns:
        image_per_batch: images. Shape: batch_size x width x height x 3
        label_per_batch: labels. Shape: batch_size x object_num
        delta_per_batch: bounding box deltas. Shape: batch_size x object_num x 4 ([dx ,dy, dw, dh])
        aidx_per_batch: index of anchors that are responsible for prediction. Shape: batch_size x object_num
        bbox_per_batch: bounding boxes. Shape: batch_size x object_num x 4 ([cx, cy, w, h])
    """

    bboxes_per_img = cPickle.load(open(data_dir + "train_bboxes_per_img.pkl"))
    img_paths = cPickle.load(open(data_dir + "train_img_paths.pkl"))

    image_per_batch = []
    label_per_batch = []
    bbox_per_batch  = []
    delta_per_batch = []
    aidx_per_batch  = []

    batch_size = 4

    for i in range(batch_size):
        img_path = img_paths[i]
        bboxes = bboxes_per_img[i]

        # load the image
        img = cv2.imread(img_path, -1).astype(np.float32)
        image_per_batch.append(img)
        orig_h, orig_w, _ = [float(v) for v in img.shape]

        # load annotations
        label_per_batch.append([b[4] for b in bboxes])
        gt_bbox = np.array([[b[0], b[1], b[2], b[3]] for b in bboxes])
        bbox_per_batch.append(gt_bbox)

        aidx_per_image, delta_per_image = [], []
        aidx_set = set()
        for i in range(len(gt_bbox)):
            overlaps = batch_iou(model.anchor_boxes, gt_bbox[i])

            aidx = len(model.anchor_boxes)
            for ov_idx in np.argsort(overlaps)[::-1]:
                if overlaps[ov_idx] <= 0:
                    break
                if ov_idx not in aidx_set:
                    aidx_set.add(ov_idx)
                    aidx = ov_idx
                    break

            if aidx == len(model.anchor_boxes):
                # even the largeset available overlap is 0, thus, choose one with the
                # smallest square distance
                dist = np.sum(np.square(gt_bbox[i] - model.anchor_boxes), axis=1)
                for dist_idx in np.argsort(dist):
                    if dist_idx not in aidx_set:
                        aidx_set.add(dist_idx)
                        aidx = dist_idx
                        break

            box_cx, box_cy, box_w, box_h = gt_bbox[i]
            delta = [0]*4
            delta[0] = (box_cx - model.anchor_boxes[aidx][0])/model.anchor_boxes[aidx][2]
            delta[1] = (box_cy - model.anchor_boxes[aidx][1])/model.anchor_boxes[aidx][3]
            delta[2] = np.log(box_w/model.anchor_boxes[aidx][2])
            delta[3] = np.log(box_h/model.anchor_boxes[aidx][3])

            aidx_per_image.append(aidx)
            delta_per_image.append(delta)

        delta_per_batch.append(delta_per_image)
        aidx_per_batch.append(aidx_per_image)

    return image_per_batch, label_per_batch, delta_per_batch, aidx_per_batch, bbox_per_batch
