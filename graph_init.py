from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow_helpers import GRU_params_init, attractor_net_init, GRU, attractor_net_loss_function
import tensorflow as tf


class GRU_attractor(object):
    def __init__(self, ops, inputs):
        X, Y, attractor_tgt_net = inputs['X'], inputs['Y'], inputs['attractor_tgt_net'] # tensor inputs

        #
        # GRAPH structuring
        #
        params = GRU_params_init(ops)
        attr_net = attractor_net_init(ops['hid'], ops['attractor_dynamics'], ops['h_hid'])
        params['attr_net'] = attr_net

        if 'pos' in ops['problem_type']:
            # pass in the embedding instead of direct index tensor
            [Y_, h_net_seq, h_attractor_collection, h_clean_seq] = GRU(X, ops, params)

            # mask out irrelevant h:
            # since all hs are (seq_len, batch, h_hid), transpose mask
            Y_transposed = tf.transpose(Y, [1,0]) # (batch_size, seq_len) -> (seq_len, batch_size)
            mask_flat = tf.cast(tf.sign(Y_transposed), dtype="bool") # 0->False, 1->True
            indices_to_collect = tf.where(mask_flat)

            # gather_nd will grab the whole row (hidden vector) since indices_to_collect is (dim-1)
            h_collected = tf.gather_nd(h_net_seq, indices_to_collect)
            h_clean_seq_collect = tf.gather_nd(h_clean_seq, indices_to_collect)

            h_clean_seq = tf.reshape(h_clean_seq_collect, [-1, ops['hid']])
            h_net_seq = tf.reshape(h_collected, [-1, ops['hid']])
        else:
            [Y_, h_net_seq, h_attractor_collection, h_clean_seq] = GRU(X, ops, params)
            h_net_seq = tf.reshape(h_net_seq, [-1, ops['hid']]) # (seq_len, batch, hid) -> (..., hid)

        #
        # LOSS
        #
        if 'pos' in ops['problem_type']:# cross entropy loss for sequence tagging
            # Y_: (batch_size, seq_len, n_classes), Y: (batch, seq_len)
            fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_, labels=Y)
        #     collapsed_class_Y = tf.reduce_max(Y, axis=2)
            mask = tf.cast(tf.sign(Y), dtype=tf.float32)
            loss_per_example_per_step = fake_loss*mask #since we only care about information with the real class
            loss_per_example_sum = tf.reduce_sum(loss_per_example_per_step, reduction_indices=[1])
            loss_per_example_average = loss_per_example_sum/tf.reduce_sum(mask, axis=[1])
            pred_loss_op = tf.reduce_mean(loss_per_example_average, name="loss")

            # reshape mask from [batch_size, seq_len] -> [batch_size*seq_len, 1], we will later
            # multiply entire hidden vectors by either 1, or 0.
            attr_loss_op, input_bias = attractor_net_loss_function(attractor_tgt_net, attr_net, ops['attractor_regularization_lambda'],
                                                                  ops['attractor_noise_level'], ops)


        else: # MSE for singular output
            pred_loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2) / .25)
            attr_loss_op, input_state = \
                   attractor_net_loss_function(attractor_tgt_net, attr_net, ops['attractor_regularization_lambda'],
                                               ops['attractor_noise_level'], ops)


        #
        # TRAINING_OPS
        #
        prediction_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TASK_WEIGHTS")
        if ops['training_mode'] == 'attractor_on_both':
            prediction_parameters += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ATTRACTOR_WEIGHTS")
        attr_net_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ATTRACTOR_WEIGHTS")

        # Define optimizer for prediction task
        optimizer_pred = tf.train.AdamOptimizer(learning_rate=0.008)
        pred_train_op = optimizer_pred.minimize(pred_loss_op, var_list=prediction_parameters)
        # Define optimizer for attractor net task
        attr_train_op = None
        if (ops['n_attractor_iterations'] > 0):
            optimizer_attr = tf.train.AdamOptimizer(learning_rate=0.008)
            attr_train_op = optimizer_attr.minimize(attr_loss_op, var_list=attr_net_parameters)

        #
        # METRICS
        #
        if 'pos' in ops['problem_type']: # apply mask, average by element nonmasekd
            id_predicted = tf.argmax(tf.nn.softmax(Y_), axis=2)
            fake_accuracy = tf.cast(tf.equal(id_predicted, Y), dtype=tf.float32)
            accuracy_masked = fake_accuracy*mask
            accuracy_per_example = tf.reduce_sum(accuracy_masked, 1)/tf.reduce_sum(mask, axis=[1])
            accuracy = tf.reduce_mean(accuracy_per_example, name="valid_accuracy")
        else:
            correct_pred = tf.equal(tf.round(Y_), Y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        self.pred_loss_op = pred_loss_op
        self.pred_train_op = pred_train_op
        self.attr_loss_op = attr_loss_op
        self.attr_train_op = attr_train_op
        self.accuracy = accuracy


        self.h_clean_seq = h_clean_seq
        self.h_net_seq = h_net_seq
        self.h_attractor_collection = h_attractor_collection
        self.output = Y_