from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow_helpers import GRU_params_init, attractor_net_init, GRU, attractor_net_loss_function
import tensorflow as tf


class GRU_attractor(object):
    def __init__(self, ops, inputs, direction='forward', suffix=''):
        if direction == 'forward':
            X, Y, attractor_tgt_net = inputs['X'], inputs['Y'], inputs['attractor_tgt_net'] # tensor inputs
        elif direction == 'backward':
            # Input is always expected as X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
            #                             Y = tf.placeholder("float", [None, N_CLASSES=1]), OR [None, SEQ_LEN])
            # attractor_tgt_net is produced by the network itself, so we don't reverse it.
            X, Y, attractor_tgt_net = tf.reverse(inputs['X'], axis=[1]), tf.reverse(inputs['Y'], axis=[1]), \
                                      inputs['attractor_tgt_net']  # tensor inputs

        #
        # GRAPH structuring
        #
        params = GRU_params_init(ops, suffix)
        attr_net = attractor_net_init(ops['hid'], ops['attractor_dynamics'], ops['h_hid'], suffix)
        params['attr_net'] = attr_net

        #
        # Masking cell outputs
        #
        if ops['prediction_type'] == 'seq':
            # pass in the embedding instead of direct index tensor
            [h_net_seq, h_attractor_collection, h_clean_seq] = GRU(X, ops, params)
            h_clean_seq_output = h_clean_seq # store before reshaping happens
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
            [h_net_seq, h_attractor_collection, h_clean_seq] = GRU(X, ops, params)
            h_clean_seq_output = h_clean_seq # store before reshaping happens
            h_net_seq = tf.reshape(h_net_seq, [-1, ops['hid']]) # (seq_len, batch, hid) -> (..., hid)

        #
        # Attractor Loss/Training op
        #
        attr_loss_op, input_bias = attractor_net_loss_function(attractor_tgt_net, attr_net,
                                                               ops['attractor_regularization_lambda'],
                                                               ops['attractor_noise_level'], ops)

        attr_net_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ATTRACTOR_WEIGHTS")


        # Define optimizer for attractor net task
        attr_train_op  = None
        if (ops['n_attractor_iterations'] > 0):
            optimizer_attr = tf.train.AdamOptimizer(learning_rate=ops['lrate'])
            attr_train_op = optimizer_attr.minimize(attr_loss_op, var_list=attr_net_parameters)




        self.attr_loss_op = attr_loss_op
        self.attr_train_op = attr_train_op

        self.h_clean_seq_flat = h_clean_seq
        self.h_net_seq_flat = h_net_seq # pure cell ouptut (before attractor was applied)
        self.h_attractor_collection = h_attractor_collection
        self.output = h_clean_seq_output
