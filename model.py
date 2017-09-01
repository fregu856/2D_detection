import numpy as np
import tensorflow as tf
import os
import cPickle

from utilities import safe_exp, bbox_transform, bbox_transform_inv

class SqueezeDet_model(object):
    """
    - DOES:
    """

    def __init__(self, model_id):
        """
        - DOES:
        """

        self.model_id = model_id

        #self.logs_dir = "/home/fregu856/segmentation/training_logs/"
        self.logs_dir = "/root/2D_detection/training_logs/"
        self.no_of_classes = 3
        self.class_weights = cPickle.load(open("data/class_weights.pkl"))

        self.initial_lr = 1e-5 # TODO! change this according to the paper
        self.decay_steps =  3000 # TODO! change this according to the paper
        self.lr_decay_rate = 0.96 # TODO! change this according to the paper
        self.img_height = 375
        self.img_width = 1242
        self.batch_size = 4

        self.anchors_per_img = 1 # TODO! # (number of anchors per image)
        self.anchors_per_gridpoint = 3 # TODO!

        self.anchor_boxes = np.zeros((self.anchors_per_img, 4)) # TODO! (fill this with all anchor x,y,w,h)

        self.exp_thresh = 2 # TODO!

        #
        self.create_model_dirs()
        #
        self.add_placeholders()
        #
        self.add_preds() # TODO! change to better name
        #
        self.add_loss_op()
        #
        self.add_train_op()

    def create_model_dirs(self):
        """
        - DOES:
        """

        self.model_dir = self.logs_dir + "model_%s" % self.model_id + "/"
        self.checkpoints_dir = self.model_dir + "checkpoints/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

    def add_placeholders(self):
        """
        - DOES:
        """

        # TODO!

        self.img_input_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, 3], # ([batch_size, img_heigth, img_width, 3])
                    name="img_input_ph")

        self.keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob_ph")

        # (tensor where an element is 1 if the corresponding anchor is responsible
        # for detecting a ground truth object and 0 otherwise)
        self.input_mask_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, 1],
                    name="input_mask_ph")

        # (tensor used to represent anchor deltas, the 4 relative coordinates
        # to transform the anchor into the "closest" ground truth bbox) # TODO! is this true?
        self.anchor_delta_input_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, 4],
                    name="anchor_delta_input_ph")

        # (tensor used to represent anchor coordinates and size) # TODO! is this true?
        self.anchor_input_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, 4],
                    name="anchor_input_ph")

        # (tensor used to represent class labels (label of the "closest" ground
        # truth bbox for each anchor)) # TODO! is this true?
        self.labels_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, self.no_of_classes],
                    name="labels_ph")

    def create_feed_dict(self, imgs_batch, drop_prob, training):
        """
        - DOES: returns a feed_dict mapping the placeholders to the actual
        input data (this is how we run the network on specific data).
        """

        # TODO! update once add_placeholers is done

        feed_dict = {}
        feed_dict[self.training_ph] = training
        feed_dict[self.drop_prob_ph] = drop_prob
        feed_dict[self.imgs_ph] = imgs_batch

        return feed_dict

    def add_preds(self):
        """
        - DOES:
        """

        # TODO!

        # (IOU between predicted and ground truth bboxes)
        self.IOUs = tf.Variable(
                    initial_value=np.zeros((self.batch_size, self.anchors_per_img)),
                    trainable=False, name="IOUs", dtype=tf.float32)

        conv_1 = self.conv_layer("conv_1", self.img_input_ph, filters=64, size=3,
                    stride=2, padding="SAME", freeze=True)
        pool_1 = self.pooling_layer("pool_1", conv_1, size=3, stride=2, padding="SAME")

        fire_2 = self.fire_layer("fire_2", pool_1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
        fire_3 = self.fire_layer("fire_3", fire_2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
        pool_3 = self.pooling_layer("pool_3", fire_3, size=3, stride=2, padding="SAME")

        fire_4 = self.fire_layer("fire_4", pool_3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
        fire_5 = self.fire_layer("fire_5", fire_4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
        pool_5 = self.pooling_layer("pool_5", fire_5, size=3, stride=2, padding="SAME")

        fire_6 = self.fire_layer("fire_6", pool_5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
        fire_7 = self.fire_layer("fire_7", fire_6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
        fire_8 = self.fire_layer("fire_8", fire_7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
        fire_9 = self.fire_layer("fire_9", fire_8, s1x1=64, e1x1=256, e3x3=256, freeze=False)

        # (two extra fire modules that are not trained before)
        fire_10 = self.fire_layer("fire_10", fire_9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
        fire_11 = self.fire_layer("fire11", fire_10, s1x1=96, e1x1=384, e3x3=384, freeze=False)
        dropout_11 = tf.nn.dropout(fire_11, self.keep_prob_ph, name="dropout_11")

        no_of_outputs = self.anchors_per_gridpoint*(self.no_of_classes + 1 + 4)
        self.preds = self.conv_layer("preds", dropout_11, filters=no_of_outputs,
                    size=3, stride=1, padding="SAME", xavier=False, relu=False,
                    stddev=0.0001)

    def add_interp_graph(self):
        # TODO! change to a better name

        preds = self.preds

        # class probabilities:
        no_of_class_probs = self.anchors_per_gridpoint*self.no_of_classes # (K*C) (total no of class probs per grid point)
        pred_class_logits = preds[:, :, :, :no_of_class_probs]
        pred_class_logits = tf.reshape(pred_class_logits, [-1, self.no_of_classes])
        pred_class_probs = tf.nn.softmax(pred_class_logits)
        pred_class_probs = tf.reshape(pred_class_probs,
                    [self.batch_size, self.anchors_per_img, self.no_of_classes])
        self.pred_class_probs = pred_class_probs

        # confidence scores:
        no_of_conf_scores = self.anchors_per_gridpoint # (total no of conf scores per grid point)
        pred_conf_scores = preds[:, :, :, no_of_class_probs:no_of_class_probs + no_of_conf_scores]
        pred_conf_scores = tf.reshape(pred_conf_scores,
                    [self.batch_size, self.anchors_per_img])
        pred_conf_scores = tf.sigmoid(pred_conf_scores) # (normalize between 0 and 1)
        self.pred_conf_scores = pred_conf_scores

        # bbox deltas: (the four numbers that describe how to transform the anchor bbox to the predicted bbox)
        pred_bbox_deltas = preds[:, :, :, no_of_class_probs + no_of_conf_scores:]
        pred_bbox_deltas = tf.reshape(pred_bbox_deltas,
                    [self.batch_size, self.anchors_per_img, 4])
        self.pred_bbox_deltas = pred_bbox_deltas

        # number of ground truth objects in the batch (used to normalize bbox and
        # classification loss):
        self.no_of_gt_objects = tf.reduce_sum(self.input_mask_ph)

        # transform the anchor bboxes to predicted bboxes using the predicted bbox deltas:
        delta_x, delta_y, delta_w, delta_h = tf.unstack(self.pred_bbox_deltas, axis=2)
        # (delta_x has shape [batch_size, anchors_per_img]) # TODO! is this true!
        anchor_x = self.anchor_boxes[:, 0]
        anchor_y = self.anchor_boxes[:, 1]
        anchor_w = self.anchor_boxes[:, 2]
        anchor_h = self.anchor_boxes[:, 3]
        # # transformation according to eq. (1) in the paper:
        bbox_center_x = anchor_x + anchor_w*delta_x
        bbox_center_y = anchor_y + anchor_h*delta_y
        bbox_width = anchor_w*safe_exp(delta_w, self.exp_thresh)
        bbox_height = anchor_h*safe_exp(delta_h, self.exp_thresh)

        # trim the predicted bboxes so that they stay within the image:
        # # get the max and min x and y coordinates for each predicted bbox: (from the predicted center coordinates and height/width. These might be outside of the image (e.g. negative or larger than the img width))
        xmin, ymin, xmax, ymax = bbox_transform([bbox_center_x, bbox_center_y,
                    bbox_width, bbox_height])
        # # limit xmin to be in [0, img_width - 1]:
        xmin = tf.minimum(tf.maximum(0.0, xmin), self.img_width - 1.0)
        # # limit ymin to be in [0, img_height - 1]:
        ymin = tf.minimum(tf.maximum(0.0, ymin), self.img_height - 1.0)
        # # limit xmax to be in [0, img_width - 1]:
        xmax = tf.maximum(tf.minimum(self.img_width - 1.0, xmax), 0.0)
        # # limit ymax to be in [0, img_height - 1]:
        ymax = tf.maximum(tf.minimum(self.img_height - 1.0, ymax), 0.0)
        # # transform the trimmed bboxes back to center/width/height format:
        cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        self.pred_bboxes = tf.transpose(tf.stack([cx, cy, w, h]), (1, 2, 0)) # (tf.stack([cx, cy, w, h], axis=2) does the same?)

    def add_loss_op(self):
        """
        - DOES: .
        """

        # TODO!

        loss = 0
        self.loss = loss

    def add_train_op(self):
        """
        - DOES: creates a training operator for minimization of the loss.
        """

        # TODO!

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate=self.initial_lr,
                    global_step=global_step, decay_steps=self.decay_steps,
                    decay_rate=self.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step) # (global_step will now automatically be incremented)


  def fire_layer(self, layer_name, input, s1x1, e1x1, e3x3, stddev=0.01, freeze=False):
        """
        - Fire layer constructor.

        Args:
          layer_name: layer name
          input: input tensor
          s1x1: number of 1x1 filters in squeeze layer.
          e1x1: number of 1x1 filters in expand layer.
          e3x3: number of 3x3 filters in expand layer.
          freeze: if true, do not train parameters in this layer.
        """

        sq1x1 = self.conv_layer(layer_name + "/squeeze1x1", input, filters=s1x1,
                    size=1, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        ex1x1 = self.conv_layer(layer_name + "/expand1x1", sq1x1, filters=e1x1,
                    size=1, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        ex3x3 = self.conv_layer(layer_name + "/expand3x3", sq1x1, filters=e3x3,
                    size=3, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        return tf.concat([ex1x1, ex3x3], 3)

    def conv_layer(self):
        # TODO!

        test = 0

    def pooling_layer(self):
        # TODO!

        test = 0

    def filter_preds(self):
        # TODO!

        test = 0
