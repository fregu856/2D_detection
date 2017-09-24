from __future__ import print_function
# NOTE! much of this code is heavily inspired by the official SqueezeDet
# implementation in github.com/BichenWuUCB/squeezeDet

import numpy as np
import tensorflow as tf
import os
import cPickle

from utilities import safe_exp, bbox_transform, bbox_transform_inv, nms

class SqueezeDet_model(object):

    def __init__(self, model_id):
        self.model_id = model_id

        self.project_dir = "/root/2D_detection/"

        self.logs_dir =  self.project_dir + "training_logs/"

        self.batch_size = 32
        self.img_height = 375
        self.img_width = 1242

        self.no_of_classes = 3
        self.class_string_to_label = {"car": 0, "pedestrian": 1, "cyclist": 2}

        # training parameters:
        self.initial_lr = 0.01
        self.decay_steps =  10000
        self.lr_decay_rate = 0.5
        self.momentum = 0.9
        self.max_grad_norm = 1.0
        self.weight_decay = 0.0001

        # set all anchor bboxes and related parameters:
        self.anchor_bboxes = self.set_anchors()
        # # (anchor_bboxes has shape [anchors_per_img, 4])
        self.anchors_per_img = len(self.anchor_bboxes)
        self.anchors_per_gridpoint = 9

        # bbox filtering parameters:
        self.top_N_detections = 64
        self.prob_thresh = 0.005
        self.nms_thresh = 0.4
        self.plot_prob_thresh = 0.7

        # other general parameters:
        self.exp_thresh = 1.0
        self.epsilon = 1e-16

        # loss coefficients:
        self.loss_coeff_class = 1.0
        self.loss_coeff_conf_pos = 75.0
        self.loss_coeff_conf_neg = 100.0
        self.loss_coeff_bbox = 5.0

        self.load_pretrained_model = True
        if self.load_pretrained_model:
            # get the weights of a pretrained SqueezeNet model:
            self.caffemodel_weights = \
                        cPickle.load(open("data/caffemodel_weights.pkl"))

        # create all dirs for storing checkpoints and other log data:
        self.create_model_dirs()

        # add placeholders to the comp. graph:
        self.add_placeholders()

        # add everything related to the model forward pass to the comp. graph:
        self.add_forward_pass()

        # process the model output and add to the comp. graph:
        self.add_processed_output()

        # compute the batch loss and add to the comp. graph:
        self.add_loss_op()

        # add a training operation (for minimizing the loss) to the comp. graph:
        self.add_train_op()

    def create_model_dirs(self):
        self.model_dir = self.logs_dir + "model_%s" % self.model_id + "/"
        self.checkpoints_dir = self.model_dir + "checkpoints/"
        self.debug_imgs_dir = self.model_dir + "imgs/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.debug_imgs_dir)

    def add_placeholders(self):
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, 3],
                    name="imgs_ph")

        self.keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob_ph")

        self.mask_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, 1],
                    name="mask_ph")
        # # (mask_ph[i, j] == 1 if anchor j is assigned to (i.e., is responsible
        # # for detecting) a ground truth bbox in batch image i, 0 otherwise)

        self.gt_deltas_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, 4],
                    name="gt_deltas_ph")
        # # (if anchor j is assigned to a ground truth bbox in batch image i,
        # # gt_deltas_ph[i, j] == [delta_x, delta_y, delta_w, delta_h] where the
        # # deltas transform anchor j into its assigned ground truth bbox via
        # # eq. (1) in the paper. Otherwise, gt_deltas_ph[i, j] == [0, 0, 0, 0])

        self.gt_bboxes_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img, 4],
                    name="gt_bboxes_ph")
        # # (if anchor j is assigned to a ground truth bbox in batch image i,
        # # gt_bboxes_ph[i, j] == [center_x, center_y, w, h] of this assigned
        # # ground truth bbox. Otherwise, gt_bboxes_ph[i, j] == [0, 0, 0, 0])

        self.class_labels_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.anchors_per_img,
                    self.no_of_classes], name="class_labels_ph")
        # # (if anchor j is assigned to a ground truth bbox in batch image i,
        # # class_labels_ph[i, j] is the onehot encoded class label of this
        # # assigned ground truth bbox. Otherwise, class_labels_ph[i, j] is all
        # # zeros)

    def create_feed_dict(self, imgs, keep_prob, mask=None, gt_deltas=None,
                         gt_bboxes=None, class_labels=None):
        # return a feed_dict mapping the placeholders to the actual input data:
        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs
        feed_dict[self.keep_prob_ph] = keep_prob
        if mask is not None: # (we have no mask during inference)
            feed_dict[self.mask_ph] = mask
        if gt_deltas is not None:
            feed_dict[self.gt_deltas_ph] = gt_deltas
        if gt_bboxes is not None:
            feed_dict[self.gt_bboxes_ph] = gt_bboxes
        if class_labels is not None:
            feed_dict[self.class_labels_ph] = class_labels

        return feed_dict

    def add_forward_pass(self):
        # (NOTE! the layer names ("conv1", "fire2" etc.) below must match
        # the names in the pretrained SqueezeNet model when using this for
        # initialization)

        conv_1 = self.conv_layer("conv1", self.imgs_ph, filters=64, size=3,
                    stride=2, padding="SAME", freeze=True)
        pool_1 = self.pooling_layer(conv_1, size=3, stride=2, padding="SAME")

        fire_2 = self.fire_layer("fire2", pool_1, s1x1=16, e1x1=64, e3x3=64)
        fire_3 = self.fire_layer("fire3", fire_2, s1x1=16, e1x1=64, e3x3=64)
        pool_3 = self.pooling_layer(fire_3, size=3, stride=2, padding="SAME")

        fire_4 = self.fire_layer("fire4", pool_3, s1x1=32, e1x1=128, e3x3=128)
        fire_5 = self.fire_layer("fire5", fire_4, s1x1=32, e1x1=128, e3x3=128)
        pool_5 = self.pooling_layer(fire_5, size=3, stride=2, padding="SAME")

        fire_6 = self.fire_layer("fire6", pool_5, s1x1=48, e1x1=192, e3x3=192)
        fire_7 = self.fire_layer("fire7", fire_6, s1x1=48, e1x1=192, e3x3=192)
        fire_8 = self.fire_layer("fire8", fire_7, s1x1=64, e1x1=256, e3x3=256)
        fire_9 = self.fire_layer("fire9", fire_8, s1x1=64, e1x1=256, e3x3=256)

        fire_10 = self.fire_layer("fire10", fire_9, s1x1=96, e1x1=384, e3x3=384)
        fire_11 = self.fire_layer("fire11", fire_10, s1x1=96, e1x1=384, e3x3=384)
        dropout_11 = tf.nn.dropout(fire_11, self.keep_prob_ph, name="dropout_11")

        no_of_outputs = self.anchors_per_gridpoint*(self.no_of_classes + 1 + 4)
        self.preds = self.conv_layer("preds", dropout_11, filters=no_of_outputs,
                    size=3, stride=1, padding="SAME", relu=False, stddev=0.0001)

    def add_processed_output(self):
        preds = self.preds

        # get all predicted class probabilities:
        # # compute the total number of predicted class probs per grid point:
        no_of_class_probs = self.anchors_per_gridpoint*self.no_of_classes
        # # get all predicted class logits:
        pred_class_logits = preds[:, :, :, :no_of_class_probs]
        pred_class_logits = tf.reshape(pred_class_logits,
                    [-1, self.no_of_classes])
        # # convert the class logits to class probs:
        pred_class_probs = tf.nn.softmax(pred_class_logits)
        pred_class_probs = tf.reshape(pred_class_probs,
                    [self.batch_size, self.anchors_per_img, self.no_of_classes])
        self.pred_class_probs = pred_class_probs

        # get all predicted confidence scores:
        # # compute the total number of predicted conf scores per grid point:
        no_of_conf_scores = self.anchors_per_gridpoint
        # # get all predicted conf scores:
        pred_conf_scores = preds[:, :, :, no_of_class_probs:no_of_class_probs + no_of_conf_scores]
        pred_conf_scores = tf.reshape(pred_conf_scores,
                    [self.batch_size, self.anchors_per_img])
        # # normalize the conf scores to lay between 0 and 1:
        pred_conf_scores = tf.sigmoid(pred_conf_scores)
        self.pred_conf_scores = pred_conf_scores

        # get all predicted bbox deltas (the four numbers that describe how to
        # transform an anchor bbox to a predicted bbox):
        pred_bbox_deltas = preds[:, :, :, no_of_class_probs + no_of_conf_scores:]
        pred_bbox_deltas = tf.reshape(pred_bbox_deltas,
                    [self.batch_size, self.anchors_per_img, 4])
        self.pred_bbox_deltas = pred_bbox_deltas

        # compute the total number of ground truth objects in the batch (used to
        # normalize the bbox and classification losses):
        self.no_of_gt_objects = tf.reduce_sum(self.mask_ph)

        # transform the anchor bboxes to predicted bboxes using the predicted
        # bbox deltas:
        delta_x, delta_y, delta_w, delta_h = tf.unstack(self.pred_bbox_deltas, axis=2)
        # # (delta_x has shape [batch_size, anchors_per_img])
        anchor_x = self.anchor_bboxes[:, 0]
        anchor_y = self.anchor_bboxes[:, 1]
        anchor_w = self.anchor_bboxes[:, 2]
        anchor_h = self.anchor_bboxes[:, 3]
        # # transformation according to eq. (1) in the paper:
        bbox_center_x = anchor_x + anchor_w*delta_x
        bbox_center_y = anchor_y + anchor_h*delta_y
        bbox_width = anchor_w*safe_exp(delta_w, self.exp_thresh)
        bbox_height = anchor_h*safe_exp(delta_h, self.exp_thresh)
        # # trim the predicted bboxes so that they stay within the image:
        # # # get the max and min x and y coordinates for each predicted bbox
        # # # from the predicted center coordinates and height/width (these
        # # # might lay outside of the image (e.g. be negative or larger than
        # # # the img width)):
        xmin, ymin, xmax, ymax = bbox_transform([bbox_center_x, bbox_center_y,
                    bbox_width, bbox_height])
        # # # limit xmin to be in [0, img_width - 1]:
        xmin = tf.minimum(tf.maximum(0.0, xmin), self.img_width - 1.0)
        # # # limit ymin to be in [0, img_height - 1]:
        ymin = tf.minimum(tf.maximum(0.0, ymin), self.img_height - 1.0)
        # # # limit xmax to be in [0, img_width - 1]:
        xmax = tf.maximum(tf.minimum(self.img_width - 1.0, xmax), 0.0)
        # # # limit ymax to be in [0, img_height - 1]:
        ymax = tf.maximum(tf.minimum(self.img_height - 1.0, ymax), 0.0)
        # # # transform the trimmed bboxes back to center/width/height format:
        cx, cy, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        self.pred_bboxes = tf.transpose(tf.stack([cx, cy, w, h]), (1, 2, 0))

        # compute the IOU between predicted and ground truth bboxes:
        pred_bboxes = bbox_transform(tf.unstack(self.pred_bboxes, axis=2))
        gt_bboxes = bbox_transform(tf.unstack(self.gt_bboxes_ph, axis=2))
        IOU = self.tensor_IOU(pred_bboxes, gt_bboxes)
        mask = tf.reshape(self.mask_ph, [self.batch_size, self.anchors_per_img])
        masked_IOU = IOU*mask
        self.IOUs = masked_IOU

        # compute Pr(class) = Pr(class | object)*Pr(object):
        probs = self.pred_class_probs*tf.reshape(self.pred_conf_scores,
                    [self.batch_size, self.anchors_per_img, 1])
        # for each predicted bbox, compute what object class it most likely
        # contains according to the model output:
        self.detection_classes = tf.argmax(probs, 2)
        # for each predicted bbox, compute the predicted probability that it
        # actually contains this most likely object class:
        self.detection_probs = tf.reduce_max(probs, 2)
        # # (self.detection_probs and self.detection_classes have shape
        # # [batch_size, anchors_per_img])

    def add_loss_op(self):
        # compute the class cross-entropy loss (adds a small value to log to
        # prevent it from blowing up):
        class_loss = (self.class_labels_ph*(-tf.log(self.pred_class_probs + self.epsilon)) +
                    (1 - self.class_labels_ph)*(-tf.log(1 - self.pred_class_probs + self.epsilon)))
        class_loss = self.loss_coeff_class*self.mask_ph*class_loss
        class_loss = tf.reduce_sum(class_loss)
        # # normalize the class loss (tf.truediv is used to ensure that we get
        # # no integer divison):
        class_loss = tf.truediv(class_loss, self.no_of_gt_objects)
        self.class_loss = class_loss
        tf.add_to_collection("losses", self.class_loss)

        # compute the confidence score regression loss (this doesn't look like
        # the conf loss in the paper, but they are actually equivalent since
        # self.IOUs is masked as well):
        input_mask = tf.reshape(self.mask_ph, [self.batch_size,
                    self.anchors_per_img])
        conf_loss_per_img = (tf.square(self.IOUs - self.pred_conf_scores)*
                    (input_mask*self.loss_coeff_conf_pos/self.no_of_gt_objects +
                    (1 - input_mask)*self.loss_coeff_conf_neg/(self.anchors_per_img - self.no_of_gt_objects)))
        conf_loss_per_img = tf.reduce_sum(conf_loss_per_img, reduction_indices=[1])
        conf_loss = tf.reduce_mean(conf_loss_per_img)
        self.conf_loss = conf_loss
        tf.add_to_collection("losses", self.conf_loss)
        # # (not sure if we're actually supposed to use reduce_mean in this loss,
        # # if so I feel like we should divide with the number of gt objects per
        # # img instead. Think this might be why self.loss_coeff_conf_pos/neg is
        # # so much larger than the other coefficients)

        # compute the bbox regression loss:
        bbox_loss = self.mask_ph*(self.pred_bbox_deltas - self.gt_deltas_ph)
        bbox_loss = self.loss_coeff_bbox*tf.square(bbox_loss)
        bbox_loss = tf.reduce_sum(bbox_loss)
        bbox_loss = tf.truediv(bbox_loss, self.no_of_gt_objects)
        self.bbox_loss = bbox_loss
        tf.add_to_collection("losses", self.bbox_loss)

        # compute the total loss by summing the above losses and all variable
        # weight decay losses:
        self.loss = tf.add_n(tf.get_collection("losses"))

    def add_train_op(self):
        # create an optimizer:
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(learning_rate=self.initial_lr,
                    global_step=global_step, decay_steps=self.decay_steps,
                    decay_rate=self.lr_decay_rate, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.momentum)

        # perform maximum clipping of the gradients:
        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        for i, (grad, var) in enumerate(grads_and_vars):
            grads_and_vars[i] = (tf.clip_by_norm(grad, self.max_grad_norm), var)

        # create the train op (global_step will automatically be incremented):
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def fire_layer(self, layer_name, input, s1x1, e1x1, e3x3, stddev=0.01, freeze=False):
        # (NOTE! the layer names ("/squeeze1x1" etc.) below must match the
        # names in the pretrained SqueezeNet model when using this for
        # initialization)

        sq1x1 = self.conv_layer(layer_name + "/squeeze1x1", input, filters=s1x1,
                    size=1, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        ex1x1 = self.conv_layer(layer_name + "/expand1x1", sq1x1, filters=e1x1,
                    size=1, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        ex3x3 = self.conv_layer(layer_name + "/expand3x3", sq1x1, filters=e3x3,
                    size=3, stride=1, padding="SAME", stddev=stddev, freeze=freeze)

        return tf.concat([ex1x1, ex3x3], 3)

    def conv_layer(self, layer_name, input, filters, size, stride, padding="SAME",
                   freeze=False, xavier=False, relu=True, stddev=0.001):
        channels = input.get_shape().as_list()[3]

        use_pretrained_params = False
        if self.load_pretrained_model:
            # get the pretrained parameter values if possible:
            cw = self.caffemodel_weights
            if layer_name in cw:
                # re-order the caffe kernel with shape [filters, channels, h, w]
                # to a tf kernel with shape [h, w, channels, filters]:
                kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
                bias_val = cw[layer_name][1]
                # check the shape:
                if kernel_val.shape == (size, size, channels, filters) and (bias_val.shape == (filters, )):
                    use_pretrained_params = True
                else:
                    print("Shape of the pretrained parameter of %s does not match, use randomly initialized parameter" % layer_name)
            else:
                print("Cannot find %s in the pretrained model. Use randomly initialized parameters" % layer_name)

        with tf.variable_scope(layer_name) as scope:
            # create the parameter initializers:
            if use_pretrained_params:
                print("Using pretrained init for " + layer_name)

                kernel_init = tf.constant(kernel_val , dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                print("Using Xavier init for " + layer_name)

                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(0.0)
            else:
                print("Using random normal init for " + layer_name)

                kernel_init = tf.truncated_normal_initializer(stddev=stddev,
                            dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            # create the variables:
            kernel = self.variable_with_weight_decay("kernel",
                        shape=[size, size, channels, filters], wd=self.weight_decay,
                        initializer=kernel_init, trainable=(not freeze))
            biases = self.get_variable("biases", shape=[filters], dtype=tf.float32,
                        initializer=bias_init, trainable=(not freeze))

            # convolution:
            conv = tf.nn.conv2d(input, kernel, strides=[1, stride, stride, 1],
                        padding=padding) + biases

            # apply ReLu if supposed to:
            if relu:
                out = tf.nn.relu(conv)
            else:
                out = conv

            return out

    def pooling_layer(self, input, size, stride, padding="SAME"):
        out = tf.nn.max_pool(input, ksize=[1, size, size, 1],
                    strides=[1, stride, stride, 1], padding=padding)

        return out

    def filter_prediction(self, boxes, probs, class_inds):
        # (boxes, probs and class_inds are lists of length anchors_per_img)

        if self.top_N_detections < len(probs):
            # get the top_N_detections largest probs and their corresponding
            # boxes and class_inds:
            # # (order[0] is the index of the largest value in probs, order[1]
            # # the index of the second largest value etc. order has length
            # # top_N_detections)
            order = probs.argsort()[:-self.top_N_detections-1:-1]
            probs = probs[order]
            boxes = boxes[order]
            class_inds = class_inds[order]
        else:
            # remove all boxes, probs and class_inds corr. to
            # prob values <= prob_thresh:
            filtered_idx = np.nonzero(probs > self.prob_thresh)[0]
            probs = probs[filtered_idx]
            boxes = boxes[filtered_idx]
            class_inds = class_inds[filtered_idx]

        final_boxes = []
        final_probs = []
        final_class_inds = []
        for c in range(self.no_of_classes):
            inds_for_c = [i for i in range(len(probs)) if class_inds[i] == c]
            keep = nms(boxes[inds_for_c], probs[inds_for_c], self.nms_thresh)
            for i in range(len(keep)):
                if keep[i]:
                  final_boxes.append(boxes[inds_for_c[i]])
                  final_probs.append(probs[inds_for_c[i]])
                  final_class_inds.append(c)

        return final_boxes, final_probs, final_class_inds

    def variable_with_weight_decay(self, name, shape, wd, initializer, trainable=True):
        var = self.get_variable(name, shape=shape, dtype=tf.float32,
                    initializer=initializer, trainable=trainable)

        if wd is not None and trainable:
            # add a variable weight decay loss:
            weight_decay = wd*tf.nn.l2_loss(var)
            tf.add_to_collection("losses", weight_decay)

        return var

    def get_variable(self, name, shape, dtype, initializer, trainable=True):
        # (this wrapper function of tf.get_variable is needed because when the
        # initializer is a constant (kernel_init = tf.constant(kernel_val,
        # dtype=tf.float32)), you should not specify the shape in
        # tf.get_variable)

        if not callable(initializer):
            var = tf.get_variable(name, dtype=dtype, initializer=initializer,
                        trainable=trainable)
        else:
            var = tf.get_variable(name, shape, dtype=dtype, initializer=initializer,
                        trainable=trainable)

        return var

    def tensor_IOU(self, box1, box2):
        # intersection:
        xmin = tf.maximum(box1[0], box2[0])
        ymin = tf.maximum(box1[1], box2[1])
        xmax = tf.minimum(box1[2], box2[2])
        ymax = tf.minimum(box1[3], box2[3])
        w = tf.maximum(0.0, xmax - xmin)
        h = tf.maximum(0.0, ymax - ymin)
        intersection_area = w*h

        # union:
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
        union_area = w1*h1 + w2*h2 - intersection_area

        IOU = intersection_area/(union_area + self.epsilon)
        # # (don't think self.epsilon is actually needed here)

        return IOU

    def set_anchors(self):
        # NOTE! this function is taken directly from
        # github.com/BichenWuUCB/squeezeDet

        H, W, B = 24, 78, 9

        anchor_shapes = np.reshape(
            [np.array(
                [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
                [ 162.,  87.], [  38.,  90.], [ 258., 173.],
                [ 224., 108.], [  78., 170.], [  72.,  43.]])]*H*W,
            (H, W, B, 2)
        )

        center_x = np.reshape(
            np.transpose(
                np.reshape(
                    np.array([np.arange(1, W+1)*float(self.img_width)/(W+1)]*H*B),
                    (B, H, W)
                ),
                (1, 2, 0)
            ),
            (H, W, B, 1)
        )

        center_y = np.reshape(
            np.transpose(
                np.reshape(
                    np.array([np.arange(1, H+1)*float(self.img_height)/(H+1)]*W*B),
                    (B, W, H)
                ),
                (2, 1, 0)
            ),
            (H, W, B, 1)
        )

        anchors = np.reshape(
            np.concatenate((center_x, center_y, anchor_shapes), axis=3),
            (-1, 4)
        )

        return anchors
