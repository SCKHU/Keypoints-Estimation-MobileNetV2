import tensorflow as tf
from layers import depthwise_separable_conv2d, conv2d, avg_pool_2d, dense, flatten, dropout, expanded_conv2d
import os
from utils import load_obj, save_obj
import numpy as np


class MobileNetV2:
    """
    MobileNet Class
    """

    def __init__(self,
                 args):

        # init parameters and input
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None
        self.args = args
        self.mean_img = None
        self.nodes = dict()

        self.pretrained_path = os.path.realpath(self.args.pretrained_path)

        self.__build()

    def __init_input(self):
        with tf.compat.v1.variable_scope('input'):
            # Input images
            self.X = tf.compat.v1.placeholder(tf.float32,
                                    [self.args.batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            # Classification supervision, it's an argmax. Feel free to change it to one-hot,
            # but don't forget to change the loss from sparse as well
            self.y = tf.compat.v1.placeholder(tf.float32, [self.args.batch_size, self.args.output_size, self.args.output_size, self.args.num_classes*3])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.compat.v1.placeholder(tf.bool)

    def __init_mean(self):
        # Preparing the mean image.
        img_mean = np.ones((1, 224, 224, 3))
        img_mean[:, :, :, 0] *= 103.939
        img_mean[:, :, :, 1] *= 116.779
        img_mean[:, :, :, 2] *= 123.68
        self.mean_img = tf.constant(img_mean, dtype=tf.float32)

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_mean()
        self.__init_input()
        self.__init_network()
        self.__init_output()

    def __expand_conv(self, name, x, num_filters, stride, k=6):
        res = False
        if stride[0] == 1 and stride[1] == 1 and num_filters == x.get_shape()[-1].value:
            if not k == None:
                res = True
        return expanded_conv2d(name, x, num_filters=num_filters, 
                    kernel_size=(3, 3), padding='SAME', stride=stride, k=k, residual=res,
                    initializer=tf.contrib.layers.xavier_initializer(), 
                    l2_strength=self.args.l2_strength, bias=None, activation=tf.nn.relu6,
                    batchnorm_enabled=self.args.batchnorm_enabled, 
                    is_training=self.is_training)

    def __init_network(self):
        with tf.compat.v1.variable_scope('MobilenetV2'):
            # Preprocessing as done in the paper
            with tf.name_scope('pre_processing'):
                preprocessed_input = (self.X - self.mean_img) / 255.0

            # Model is here!
            conv = conv2d('Conv', preprocessed_input, num_filters=int(round(32 * self.args.width_multiplier)),
                             kernel_size=(3, 3), padding='SAME', stride=(2, 2), activation=tf.nn.relu6,
                             batchnorm_enabled=self.args.batchnorm_enabled,
                             is_training=self.is_training, l2_strength=self.args.l2_strength, bias=self.args.bias)
            # 112 x 112 x 32
            conv0 = self.__expand_conv('expanded_conv_0', conv, num_filters=16, stride=(1, 1), k=None)
            conv1 = self.__expand_conv('expanded_conv_1', conv0, num_filters=24, stride=(2, 2))
            # 56 x 56 x 24
            conv2 = self.__expand_conv('expanded_conv_2', conv1, num_filters=24, stride=(1, 1))

            conv3 = self.__expand_conv('expanded_conv_3', conv2, num_filters=32, stride=(2, 2))
            # 28 x 28 x 32
            conv4 = self.__expand_conv('expanded_conv_4', conv3, num_filters=32, stride=(1, 1))
            # 28 x 28 x 32
            conv5 = self.__expand_conv('expanded_conv_5', conv4, num_filters=32, stride=(1, 1))
            # 28 x 28 x 32

            conv6 = self.__expand_conv('expanded_conv_6', conv5, num_filters=64, stride=(2, 2))
            # 14 x 14 x 64
            conv7 = self.__expand_conv('expanded_conv_7', conv6, num_filters=64, stride=(1, 1))
            # 14 x 14 x 64
            conv8 = self.__expand_conv('expanded_conv_8', conv7, num_filters=64, stride=(1, 1))
            # 14 x 14 x 64
            conv9 = self.__expand_conv('expanded_conv_9', conv8, num_filters=64, stride=(1, 1))
            # 14 x 14 x 64

            conv10 = self.__expand_conv('expanded_conv_10', conv9, num_filters=96, stride=(1, 1))
            # 14 x 14 x 96
            conv11 = self.__expand_conv('expanded_conv_11', conv10, num_filters=96, stride=(1, 1))
            # 14 x 14 x 96
            conv12 = self.__expand_conv('expanded_conv_12', conv11, num_filters=96, stride=(1, 1))
            # 14 x 14 x 96

            conv13 = self.__expand_conv('expanded_conv_13', conv12, num_filters=160, stride=(2, 2))
            # 7 x 7 x 160 
            conv14 = self.__expand_conv('expanded_conv_14', conv13, num_filters=160, stride=(1, 1))
            conv15 = self.__expand_conv('expanded_conv_15', conv14, num_filters=160, stride=(1, 1))
            conv16 = self.__expand_conv('expanded_conv_16', conv15, num_filters=320, stride=(1, 1))


            conv17 = conv2d('Conv_1', conv16, num_filters=1280,kernel_size=(1, 1),
                             padding='SAME', stride=(1, 1), activation=tf.nn.relu6,
                             batchnorm_enabled=self.args.batchnorm_enabled,
                             is_training=self.is_training, 
                             l2_strength=self.args.l2_strength, bias=self.args.bias)

            self.__add_to_nodes([conv, conv0, conv1, conv2, conv3, conv4, 
                        conv5, conv6, conv7, conv8, conv9, 
                        conv10, conv11, conv12, conv13, conv14, 
                        conv15, conv16, conv17])
            ############################################################################################

        with tf.compat.v1.variable_scope('Heatmap'):
            
            conv18 = conv2d('conv_1', conv17,
                           num_filters=self.args.num_classes*3, kernel_size=(1, 1), padding='SAME', stride=(1, 1),
                           l2_strength=self.args.l2_strength, bias=self.args.bias,
                           activation=None, batchnorm_enabled=self.args.batchnorm_enabled,
                           is_training=self.is_training)
            # 7 x 7 x 3K
            up19 = tf.concat([tf.image.resize_bilinear( conv18, (14,14) ), conv12], -1)
            conv20_dw, conv20_pw = depthwise_separable_conv2d('conv_ds_1', up19,
                                    width_multiplier=self.args.width_multiplier,
                                    num_filters=int(round(self.args.num_classes*3 * self.args.width_multiplier)), 
                                    kernel_size=(3, 3), padding='SAME',
                                    stride=(1, 1),
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    activation=None,
                                    is_training=self.is_training,
                                    l2_strength=self.args.l2_strength,
                                    biases=(self.args.bias, self.args.bias))
           
            # 28 x 28 x 3K

            up21 = tf.image.resize_bilinear( conv20_pw, (28,28) )
            up21_1 = tf.concat([tf.slice(up21, [0,0,0,0],                         [-1,-1,-1,self.args.num_classes]), conv5], -1)
            up21_2 = tf.concat([tf.slice(up21, [0,0,0,  self.args.num_classes],  [-1,-1,-1, self.args.num_classes*2]), conv5], -1)
            
            _, heatmap = depthwise_separable_conv2d('conv_ds_2_1', up21_1,
                                    width_multiplier=self.args.width_multiplier,
                                    num_filters=int(round(self.args.num_classes * self.args.width_multiplier)), 
                                    kernel_size=(3, 3), padding='SAME',
                                    stride=(1, 1),
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    activation=tf.nn.sigmoid,
                                    is_training=self.is_training,
                                    l2_strength=self.args.l2_strength,
                                    biases=(self.args.bias, self.args.bias))

            _, offset = depthwise_separable_conv2d('conv_ds_2_2', up21_2,
                                    width_multiplier=self.args.width_multiplier,
                                    num_filters=int(round(self.args.num_classes * 2 * self.args.width_multiplier)), 
                                    kernel_size=(3, 3), padding='SAME',
                                    stride=(1, 1),
                                    batchnorm_enabled=self.args.batchnorm_enabled,
                                    activation=tf.nn.tanh,
                                    is_training=self.is_training,
                                    l2_strength=self.args.l2_strength,
                                    biases=(self.args.bias, self.args.bias))            
            # 56 x 56 x 3K
            
            self.__add_to_nodes([heatmap, offset]) 
            self.heatmap  = heatmap
            self.offset = offset 
            #if self.freeze:
            #    self.preds = tf.transpose(tf.concat([heatmap, offset],-1),
            #     perm=[0, 3, 1, 2]) 
            #else:
            self.preds = (heatmap, offset)

    def __init_output(self):
        with tf.compat.v1.variable_scope('output'):
            label_heatmap  = tf.slice(self.y,                      [0,0,0,0],  [-1,-1,-1,self.args.num_classes])
            label_offset = tf.slice(self.y,[0,0,0,  self.args.num_classes],  [-1,-1,-1,self.args.num_classes*2])

            self.regularization_loss = tf.reduce_sum(tf.compat.v1.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.map_loss = tf.compat.v1.losses.mean_squared_error(labels=label_heatmap, predictions=self.heatmap)

            diff_dist = tf.square(label_offset - self.offset)
            self.offset_loss = tf.reduce_mean(tf.multiply(diff_dist, tf.concat([label_heatmap, label_heatmap], -1)))
    
            self.accuracy = self.offset_loss + self.map_loss*5
            self.loss = self.regularization_loss + self.accuracy

            # Important for Batch Normalization
            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

        # Summaries needed for TensorBoard
        with tf.name_scope('train-summary-per-iteration'):
            tf.compat.v1.summary.scalar('loss', self.loss)
            tf.compat.v1.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.compat.v1.summary.merge_all()

    def __restore(self, file_name, sess):
        try:
            print("Loading ImageNet pretrained weights...")
            variables = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobilenet_encoder')
            dict = load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            sess.run(run_list)
            print("ImageNet Pretrained Weights Loaded Initially\n\n")
        except:
            print("No pretrained ImageNet weights exist. Skipping...\n\n")

    def load_pretrained_weights(self, sess):
        self.__restore(self.pretrained_path, sess)

    def __add_to_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.name] = node

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.compat.v1.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.compat.v1.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.compat.v1.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.compat.v1.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
