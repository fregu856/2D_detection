import numpy as np
import tensorflow as tf
import os
import cPickle

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

        self.no_of_anchors = 1 # TODO!
        self.anchors_per_gridpoint = 3 # TODO!

        #
        self.create_model_dirs()
        #
        self.add_placeholders()
        #
        self.add_preds() # TODO! change to better name, might have to add more functions too
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
                    shape=[self.batch_size, self.no_of_anchors, 1],
                    name="input_mask_ph")

        # (tensor used to represent anchor deltas, the 4 relative coordinates
        # to transform the anchor into the "closest" ground truth bbox) # TODO! is this true?
        self.anchor_delta_input_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.no_of_anchors, 4],
                    name="anchor_delta_input_ph")

        # (tensor used to represent anchor coordinates and size) # TODO! is this true?
        self.anchor_input_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.no_of_anchors, 4],
                    name="anchor_input_ph")

        # (tensor used to represent class labels (label of the "closest" ground
        # truth bbox for each anchor)) # TODO! is this true?
        self.labels_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.no_of_anchors, self.no_of_classes],
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
                    initial_value=np.zeros((self.batch_size, self.no_of_anchors)),
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
        # TODO!

        preds = self.preds

        # class probabilities:
        no_of_class_probs = self.anchors_per_gridpoint*self.no_of_classes
        self.pred_class_probs = tf.reshape( tf.nn.softmax( tf.reshape( preds[:, :, :, :no_of_class_probs], [-1, mc.CLASSES] ) ), [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES])

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
