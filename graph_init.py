from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tensorflow_helpers import GRU_params_init, attractor_net_init, GRU, attractor_net_loss_function
import tensorflow as tf


class GRU_attractor(object):
    def __init__(self, ops, inputs, direction='forward', suffix=''):
        """
        inputs['mask'] - has to be 1 or 0
        """
        if direction == 'forward':
            X, mask_phldr, attractor_tgt_net = inputs['X'], inputs['mask'], inputs['attractor_tgt_net'] # tensor inputs
        elif direction == 'backward':
            # Input is always expected as X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
            #                             Y = tf.placeholder("float", [None, N_CLASSES=1]), OR [None, SEQ_LEN])
            # attractor_tgt_net is produced by the network itself, so we don't reverse it.
            X, mask_phldr, attractor_tgt_net = tf.reverse(inputs['X'], axis=[1]), tf.reverse(inputs['mask'], axis=[1]), \
                                      inputs['attractor_tgt_net']  # tensor inputs

        batch_size = tf.shape(X)[0]

        #
        # GRAPH structuring
        #
        params = GRU_params_init(ops, suffix)
        attr_net = attractor_net_init(ops['hid'], ops['attractor_dynamics'], ops['h_hid'], suffix)
        params['attr_net'] = attr_net

        #
        # Masking cell outputs
        #
        if ops['masking']:
            # pass in the embedding instead of direct index tensor
            [h_net_seq, h_attractor_collection, h_clean_seq] = GRU(X, ops, params)
            h_clean_seq_output = h_clean_seq # store before reshaping happens
            # mask out irrelevant h:
            # since all hs are (seq_len, batch, h_hid), transpose mask
            Y_transposed = tf.transpose(mask_phldr, [1,0]) # (batch_size, seq_len) -> (seq_len, batch_size)

            mask_flat = tf.cast(tf.sign(Y_transposed), dtype="bool") # 0->False, 1->True
            indices_to_collect = tf.where(mask_flat)

            # gather_nd will grab the whole row (hidden vector) since indices_to_collect is (dim-1)
            h_net_collect = tf.gather_nd(h_net_seq, indices_to_collect)
            h_clean_seq_collect = tf.gather_nd(h_clean_seq, indices_to_collect)


            h_clean_seq = tf.reshape(h_clean_seq_collect, [-1, ops['hid']])
            h_net_seq = tf.reshape(h_net_collect, [-1, ops['hid']])

            if ops['record_mutual_information']:
                h_attractor_collection_new = []
                for h in h_attractor_collection:
                    h_attractor_collected = tf.gather_nd(h, indices_to_collect)
                    h_attractor_collection_new.append(tf.reshape(h_attractor_collected, [-1, ops['h_hid']])) # note h_hid, not hid
                h_attractor_collection = h_attractor_collection_new
        else:
            [h_net_seq, h_attractor_collection, h_clean_seq] = GRU(X, ops, params)
            h_clean_seq_output = h_clean_seq # store before reshaping happens

            h_clean_seq = tf.reshape(h_clean_seq, [-1, ops['hid']])
            h_net_seq = tf.reshape(h_net_seq, [-1, ops['hid']])

            if ops['record_mutual_information']:
                h_attractor_collection_new = []
                for h in h_attractor_collection:
                    h_attractor_collection_new.append(
                        tf.reshape(h, [-1, ops['h_hid']]))  # note h_hid, not hid
                h_attractor_collection = h_attractor_collection_new

        #
        # Attractor Loss/Training op
        #
        attr_loss_op, input_bias = attractor_net_loss_function(attractor_tgt_net, attr_net,
                                                               ops['attractor_regularization_lambda'],
                                                               ops['attractor_noise_level'], ops)

        attr_net_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ATTRACTOR_WEIGHTS" + '/{}'.format(suffix))
        # Define optimizer for attractor net task
        attr_train_op  = None
        if (ops['n_attractor_iterations'] > 0):
            optimizer_attr = tf.train.AdamOptimizer(learning_rate=ops['lrate'])
            attr_train_op = optimizer_attr.minimize(attr_loss_op, var_list=attr_net_parameters)

        self.attr_loss_op = attr_loss_op
        self.attr_train_op = attr_train_op

        self.h_net_seq_flat = h_net_seq # pure cell ouptut (before attractor was applied)
        self.h_attractor_collection = h_attractor_collection
        self.h_clean_seq_flat = h_clean_seq
        if ops['prediction_type'] == 'final' and ops['masking']:
            # if we only need the final output and the sequences required masking do this:

            # forward: get index right before first zero element
            # backward: take the last element since they are all aligned
            if direction == 'forward':
                Y_transposed_plus_zeroes = tf.concat([Y_transposed, tf.cast(tf.zeros([1, batch_size]),'int64')], axis=0) # since full sequences will get ignored othewise, force append with 0
                last_element_id = tf.argmin(Y_transposed_plus_zeroes, axis=0) - 1  # find first zero'th element across sequences and take one right before it.
                collect_indices = tf.stack([last_element_id, tf.cast(tf.range(batch_size), 'int64')], axis=1) #(batch_size, 2). Tuples of format: [sequence_position, batch_id]
                collect = tf.gather_nd(h_clean_seq_output, collect_indices)
            else:
                # h_clean_seq_output: (seq_len, batch_size, n_hid)
                # get the last element slice
                slice_shape = [1, -1, ops['hid']]  # get onle 1 sequence element, all batches, all hidden units
                # start at last element (ops['seq_len']), first batch element, first hidden unit:
                collect = tf.slice(h_clean_seq_output, [ops['seq_len']-1, 0, 0], slice_shape) #seq_len - 1 since we need id, not actual count
                collect = tf.squeeze(collect, axis=0)

            self.output = collect # (batch_size, n_hid)
        else:
            self.output = h_clean_seq_output
