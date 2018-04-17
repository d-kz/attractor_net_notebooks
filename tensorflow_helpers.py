import tensorflow as tf
import numpy as np
from helper_functions import get_batches


#
# SMALL STUFF:
#
################ mozer_get_variable #####################################################
def mozer_get_variable(vname, mat_dim):
    if (len(mat_dim) == 1): # bias
        val = 0.1 * tf.random_normal(mat_dim)
        var = tf.get_variable(vname, initializer=val)

    else:
        #var = tf.get_variable(vname, shape=mat_dim,
        #                    initializer=tf.contrib.layers.xavier_initializer())

        #val = tf.random_normal(mat_dim)
        #var = tf.get_variable(vname, initializer=val)

        val = tf.random_normal(mat_dim)
        val = 2 * val / tf.reduce_sum(tf.abs(val),axis=0, keep_dims=True)
        var = tf.get_variable(vname, initializer=val)
    return var


def batch_tensor_collect(sess, input_tensors, X, Y, X_data, Y_data, batch_size):
    batches = get_batches(batch_size, X_data, Y_data)
    collect_outputs = [[] for i in range(len(input_tensors))]
    for (batch_x, batch_y) in batches:
        outputs = sess.run(input_tensors, feed_dict={X: batch_x, Y: batch_y})
        for i, output in enumerate(outputs):
            collect_outputs[i].append(output)

    # merge all
    for i in range(len(input_tensors)):
        output = np.array(collect_outputs[i])
        # import pdb;pdb.set_trace()
        if len(output[0].flatten()) > 1: # for actual tensor collections, merge batches
            output = np.concatenate(output, axis=0)
        else: # for just values, find the average
            output = np.mean(output)
        collect_outputs[i] = output

    return collect_outputs



#
# GRAPH HELPER FUNCTIONS
#
######################################################################################
def task_loss(Y, Y_, ops):
    if ops['prediction_type'] == 'seq':# cross entropy loss for sequence tagging
        # Y_: (batch_size, seq_len, n_classes), Y: (batch, seq_len)
        fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_, labels=Y)
        # collapsed_class_Y = tf.reduce_max(Y, axis=2)
        mask = tf.cast(tf.sign(Y), dtype=tf.float32)
        loss_per_example_per_step = fake_loss*mask #since we only care about information with the real class
        loss_per_example_sum = tf.reduce_sum(loss_per_example_per_step, reduction_indices=[1])
        loss_per_example_average = loss_per_example_sum/tf.reduce_sum(mask, axis=[1])
        pred_loss_op = tf.reduce_mean(loss_per_example_average, name="loss")
    else: # MSE for singular output
        pred_loss_op = tf.reduce_mean(tf.pow(Y_ - tf.cast(Y, 'float'), 2) / .25)
    return pred_loss_op

def task_accuracy(Y, Y_, ops):
    """
    :param Y: [batch_size, length OR output=1]
    :param Y_: prediction [batch_size, length, classes_logits]
    :param ops: model params
    :return: accuracy tensor operation
    """
    if ops['prediction_type'] == 'seq':  # apply mask, average by element nonmasekd
        mask = tf.cast(tf.sign(Y), dtype=tf.float32)
        id_predicted = tf.argmax(tf.nn.softmax(Y_), axis=2)
        fake_accuracy = tf.cast(tf.equal(id_predicted, Y), dtype=tf.float32)
        accuracy_masked = fake_accuracy * mask
        accuracy_per_example = tf.reduce_sum(accuracy_masked, 1) / tf.reduce_sum(mask, axis=[1])
        accuracy = tf.reduce_mean(accuracy_per_example, name="valid_accuracy")
    else:
        correct_pred = tf.equal(tf.cast(tf.round(Y_), tf.int32), tf.cast(Y, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def project_init(in_size, out_size, reuse):
    if reuse:
        scope = tf.variable_scope("TASK_WEIGHTS", reuse=True)
    else:
        scope = tf.variable_scope("TASK_WEIGHTS")
    with scope:
        b_out = mozer_get_variable("b_out", [out_size])
        W_out = mozer_get_variable("W_out", [in_size, out_size])
    return W_out, b_out

def project_into_output(Y, output, in_size, out_size, ops,reuse=False):
    W_out, b_out = project_init(in_size, out_size, reuse)

    batch_size = tf.shape(Y)[0]
    if ops['prediction_type'] == 'seq':
        # for efficiency's sake just do one matmul.
        output_trans = tf.transpose(output, [1, 0, 2])  # [seq_len, batch_size, n_hid] -> [batch_size, seq_len, n_hid]
        output_trans = tf.reshape(output_trans,
                                  [-1, in_size])  # [batch_size, seq_len, n_hid]-> [-1, n_hid]
        out = tf.nn.sigmoid(tf.matmul(output_trans, W_out) + b_out)
        Y_ = tf.reshape(out, [batch_size, ops['seq_len'], out_size])
    elif ops['masking']:
        Y_ = tf.nn.sigmoid(tf.matmul(output, W_out) + b_out)
    else:
        # without masking we had no need to bother with dimensions, so let's just leave it as is. (sorry)
        Y_ = tf.nn.sigmoid(tf.matmul(output[-1], W_out) + b_out)
    return Y_


"""
Computes the F1 score on BIO tagged data

@author: Nils Reimers
"""


def compute_f1(predictions, correct, idx2Label):
    label_pred = [idx2Label[element] for element in predictions]

    label_correct = [idx2Label[element] for element in correct]
    # print(label_correct[0:1000], label_pred[0:1000])
    # print label_pred
    # print label_correct

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    # for sentenceIdx in range(len(guessed_sentences)):
    guessed = guessed_sentences
    correct = correct_sentences
    assert (len(guessed) == len(correct))
    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == 'B':  # A new chunk starts
            count += 1
            if guessed[idx] == correct[idx]:
                # print(guessed[idx:idx + 5], correct[idx:idx + 5])

                idx += 1
                correctlyFound = True



                while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == 'I':  # The chunk in correct was longer
                        correctlyFound = False

                if correctlyFound:
                    print(guessed[idx-5:idx], correct[idx-5:idx])
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision

#
# ATTRACTOR NETWORK:
#
############### RUN_ATTRACTOR_NET #################################################
def run_attractor_net(input_bias, attr_net, ops):
    # input_bias - nonsquashed hidden state
    a_clean_collection = []  # for mutual inf estimation

    if (ops['n_attractor_iterations'] > 0):
        if ops['attractor_dynamics'] == 'projection2':
            # task -> attractor space
            input_bias = tf.matmul(input_bias, attr_net['W_in']) + attr_net['b']

            a_clean = tf.zeros(tf.shape(input_bias))
            for i in range(ops['n_attractor_iterations']):
                # my version:
                #                 a_clean = tf.tanh(tf.matmul(a_clean, attr_net['Wconstr']) \
                #                           +  input_bias #attr_net['scale'] *
                a_clean = tf.matmul(tf.tanh(a_clean), attr_net['Wconstr']) \
                          + input_bias
                a_clean_collection.append(a_clean)

            # attractor space -> task
            a_clean = tf.tanh(tf.matmul(a_clean, attr_net['W_out']) + attr_net['b_out'])
        elif ops['attractor_dynamics'] == 'projection3':
            # task -> attractor space
            h_bias = tf.matmul(tf.tanh(input_bias), attr_net['W_in']) + attr_net['b']

            a_clean = tf.zeros(tf.shape(h_bias))
            for i in range(ops['n_attractor_iterations']):
                a_clean = tf.tanh(tf.matmul(a_clean, attr_net['Wconstr']) + h_bias)
                a_clean_collection.append(a_clean)

            # attractor space -> tasky
            a_clean = tf.tanh(tf.matmul(a_clean, attr_net['W_out']) + attr_net['b_out'] + input_bias)
        else:
            a_clean = tf.zeros(tf.shape(input_bias))
            for i in range(ops['n_attractor_iterations']):
                a_clean = tf.matmul(tf.tanh(a_clean), attr_net['Wconstr']) \
                          + attr_net['scale'] * input_bias + attr_net['b']
                a_clean_collection.append(a_clean)
            # a_clean = tf.tanh(tf.matmul(a_clean, attr_net['Wconstr']) \
            #                                   +  attr_net['scale'] * input_bias + attr_net['b'])

            a_clean = tf.tanh(a_clean)
    else:
        a_clean = tf.tanh(input_bias)
    return a_clean, a_clean_collection


############### ATTRACTOR NET LOSS FUNCTION #####################################

def attractor_net_loss_function(attractor_tgt_net, attr_net, regularization_strength, noise_level_added, ops):
    # attractor_tgt_net has dimensions #examples X #hidden
    #                   where the target value is tanh(attractor_tgt_net)

    # clean-up for attractor net training
    if (ops['attractor_noise_level'] >= 0.0):  # Gaussian mean-zero noise
        input_bias = attractor_tgt_net + noise_level_added \
                                         * tf.random_normal(tf.shape(attractor_tgt_net))
    else:  # Bernoulli dropout
        input_bias = attractor_tgt_net * \
                     tf.cast((tf.random_uniform(tf.shape(attractor_tgt_net)) \
                              >= -noise_level_added), tf.float32)

    a_cleaned, _ = run_attractor_net(input_bias, attr_net, ops)

    # loss is % reduction in noise level
    attr_tgt = tf.tanh(attractor_tgt_net)
    attr_loss = tf.reduce_mean(tf.pow(attr_tgt - a_cleaned, 2)) / \
                tf.reduce_mean(tf.pow(attr_tgt - tf.tanh(input_bias), 2))

    if ops['attractor_regularization'] == 'l2_regularization':
        print("L2 reg-n")
        attr_loss += regularization_strength * tf.nn.l2_loss(attr_net['W'])
    elif ops['attractor_regularization'] == 'l2_norm':
        print("L2 norm")
        attr_loss += regularization_strength * tf.norm(attr_net['W'], ord=2)
    else:
        print("No Regularization")

    return attr_loss, input_bias


def attractor_net_init(N_HIDDEN, ATTRACTOR_TYPE, N_H_HIDDEN, suffix='', reuse=False):
    # attr net weights
    # NOTE: i tried setting attractor_W = attractor_b = 0 and attractor_scale=1.0
    # which is the default "no attractor" model, but that doesn't learn as well as

    with tf.variable_scope("ATTRACTOR_WEIGHTS"):
        if reuse:
            scope = tf.variable_scope(suffix, reuse=True)
        else:
            scope = tf.variable_scope(suffix)
        with scope:
            attr_net = {}
            if ATTRACTOR_TYPE == 'projection2' or ATTRACTOR_TYPE == "projection3":  # attractor net 2
                attr_net['W_in'] = tf.get_variable("attractor_W_in", initializer=tf.eye(N_HIDDEN, num_columns=N_H_HIDDEN) +
                                                                                 .01 * tf.random_normal(
                                                                                     [N_HIDDEN, N_H_HIDDEN]))
                attr_net['W_out'] = tf.get_variable("attractor_Wout", initializer=tf.eye(N_H_HIDDEN, num_columns=N_HIDDEN) +
                                                                                  .01 * tf.random_normal(
                                                                                      [N_H_HIDDEN, N_HIDDEN]))
                attr_net['b_out'] = mozer_get_variable("attractor_b_out", [N_HIDDEN])
                attr_net['W'] = tf.get_variable("attractor_W", initializer=.01 * tf.random_normal([N_H_HIDDEN, N_H_HIDDEN]))
                attr_net['b'] = tf.get_variable("attractor_b", initializer=.01 * tf.random_normal([N_H_HIDDEN]))
            else:
                attr_net = {
                    'W': tf.get_variable("attractor_W", initializer=.01 * tf.random_normal([N_HIDDEN, N_HIDDEN])),
                    'b': tf.get_variable("attractor_b", initializer=.01 * tf.random_normal([N_HIDDEN]))
                }
            attr_net['scale'] = tf.get_variable("attractor_scale", initializer=.01 * tf.ones([1]))

    if reuse:
        scope = tf.variable_scope(suffix, reuse=True)
    else:
        scope = tf.variable_scope(suffix)
    with scope:
        # if ATTR_WEIGHT_CONSTRAINTS:  # symmetric + nonnegative diagonal weight matrix
        Wdiag = tf.matrix_band_part(attr_net['W'], 0, 0)  # diagonal
        Wlowdiag = tf.matrix_band_part(attr_net['W'], -1, 0) - Wdiag  # lower diagonal
        # the normalization will happen here automatically since we defined it as a TF op
        attr_net['Wconstr'] = Wlowdiag + tf.transpose(Wlowdiag) + tf.abs(Wdiag)
        # attr_net['Wconstr'] = .5 * (attr_net['W'] + tf.transpose(attr_net['W'])) * \
        #                      (1.0-tf.eye(N_HIDDEN)) + tf.abs(tf.matrix_band_part(attr_net['W'],0,0))

        # else:
        #     attr_net['Wconstr'] = attr_net['W']
    return attr_net

#
# GRU
#
############### GRU ###############################################################
def GRU_params_init(ops, suffix=''):
    N_INPUT = ops['in']
    N_HIDDEN = ops['hid']
    # N_CLASSES = ops['out']
    with tf.variable_scope("TASK_WEIGHTS"):
        with tf.variable_scope(suffix):
            W = {#'out': mozer_get_variable("W_out", [N_HIDDEN, N_CLASSES]),
                 'in_stack': mozer_get_variable("W_in_stack", [N_INPUT, 3*N_HIDDEN]),
                 'rec_stack': mozer_get_variable("W_rec_stack", [N_HIDDEN,3*N_HIDDEN]),
                }

            b = {#'out': mozer_get_variable("b_out", [N_CLASSES]),
                 'stack': mozer_get_variable("b_stack", [3 * N_HIDDEN]),
                }

    params = {
        'W': W,
        'b': b
    }
    return params

def GRU(X, ops, params):
    with tf.variable_scope("GRU"):
        W = params['W']
        b = params['b']
        attr_net = params['attr_net']
        N_HIDDEN = ops['hid']
        block_size = [-1, N_HIDDEN]

        def _step(accumulated_vars, input_vars):
            h_prev, _, _ = accumulated_vars
            x = input_vars

            preact = tf.matmul(x, W['in_stack'][:,:N_HIDDEN*2]) + \
                     tf.matmul(h_prev, W['rec_stack'][:,:N_HIDDEN*2]) + \
                     b['stack'][:N_HIDDEN*2]
            z = tf.sigmoid(tf.slice(preact, [0, 0 * N_HIDDEN], block_size))
            r = tf.sigmoid(tf.slice(preact, [0, 1 * N_HIDDEN], block_size))
            # new potential candidate for memory vector
            c_cand = tf.tanh( tf.matmul(x, W['in_stack'][:,N_HIDDEN*2:]) + \
                              tf.matmul(h_prev * r, W['rec_stack'][:,N_HIDDEN*2:]) + \
                              b['stack'][N_HIDDEN*2:])
            h = z * h_prev + (1.0 - z) * c_cand

            # insert attractor net
            h_net = tf.atanh(tf.minimum(.99999, tf.maximum(-.99999, h)))
            h_cleaned, h_attractor_collection = run_attractor_net(h_net, attr_net, ops)
            return [h_cleaned, h_net, h_attractor_collection]

        # X:                       (batch_size, SEQ_LEN, N_HIDDEN)
        # expected shape for scan: (SEQ_LEN, batch_size, N_HIDDEN)
        batch_size = tf.shape(X)[0]
        [h_clean_seq, h_net_seq, h_attractor_collection] = tf.scan(_step,
                  elems=tf.transpose(X, [1, 0, 2]),
                  initializer=[tf.zeros([batch_size, N_HIDDEN], tf.float32),  # h_clean
                               tf.zeros([batch_size, N_HIDDEN], tf.float32),  # h_net
                                [tf.zeros([batch_size, ops['h_hid']], tf.float32) for i in range(ops['n_attractor_iterations'])] ],
                                  name='GRU/scan')


        # if 'pos' in ops['problem_type']:
        #     # for efficiency's sake just do one matmul.
        #     h_clean_seq_trans = tf.transpose(h_clean_seq, [1,0,2]) # [seq_len, batch_size, n_hid] -> [batch_size, seq_len, n_hid]
        #     h_clean_seq_trans = tf.reshape(h_clean_seq_trans, [-1, N_HIDDEN])  # [batch_size, seq_len, n_hid]-> [-1, n_hid]
        #     out = tf.nn.sigmoid(tf.matmul(h_clean_seq_trans, W['out']) + b['out'])
        #     out = tf.reshape(out, [batch_size, ops['seq_len'], ops['out']])
        # else:
        #     out = tf.nn.sigmoid(tf.matmul(h_clean_seq[-1], W['out']) + b['out'])
        # h_clean_seq - true output of the cell
        # h_net_seq - output before attractor is applied
        return [h_net_seq, h_attractor_collection, h_clean_seq]


######### BEGIN TANH RNN ########################################################

def RNN_tanh_params_init(ops, suffix='', reuse=False):
    N_INPUT = ops['in']
    N_HIDDEN = ops['hid']
    if reuse:
        scope = tf.variable_scope("TASK_WEIGHTS", reuse=True)
    else:
        scope = tf.variable_scope("TASK_WEIGHTS")
    with scope:
        with tf.variable_scope(suffix):
            W = {'in': mozer_get_variable("W_in", [N_INPUT, N_HIDDEN]),
                 'rec': mozer_get_variable("W_rec", [N_HIDDEN, N_HIDDEN])
                 }
            b = {'rec': mozer_get_variable("b_rec", [N_HIDDEN]),
                 }

    params = {
        'W': W,
        'b': b
    }
    return params


def RNN_tanh(X, ops, params):
    with tf.variable_scope("TANH"):
        W = params['W']
        b = params['b']
        attr_net = params['attr_net']
        N_HIDDEN = ops['hid']
        block_size = [-1, N_HIDDEN]

        def _step(accumulated_vars, input_vars):
            h_prev, _, _ = accumulated_vars
            x = input_vars

            # update the hidden state but don't apply the squashing function
            h_net = tf.matmul(h_prev, W['rec']) + tf.matmul(x, W['in']) + b['rec']

            # insert attractor net
            # h_net = tf.atanh(tf.minimum(.99999, tf.maximum(-.99999, h)))
            h_cleaned, h_attractor_collection = run_attractor_net(h_net, attr_net, ops)

            return [h_cleaned, h_net, h_attractor_collection]

        # X:                       (batch_size, SEQ_LEN, N_INPUT)
        # expected shape for scan: (SEQ_LEN, batch_size, N_INPUT)
        batch_size = tf.shape(X)[0]
        [h_clean_seq, h_net_seq, h_attractor_collection] = tf.scan(_step,
                                                                   elems=tf.transpose(X, [1, 0, 2]),
                                                                   initializer=[
                                                                       tf.zeros([batch_size, N_HIDDEN], tf.float32),
                                                                       # h_clean
                                                                       tf.zeros([batch_size, N_HIDDEN], tf.float32),
                                                                       # h_net
                                                                       [tf.zeros([batch_size, ops['h_hid']], tf.float32)
                                                                        for i in range(ops['n_attractor_iterations'])]],
                                                                   name='TANH/scan')

        return [h_net_seq, h_attractor_collection, h_clean_seq]