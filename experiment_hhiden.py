#!/usr/local/bin/python

# This version of the code trains the attractor connections with a separate
# objective function than the objective function used to train all other weights
# in the network (on the prediction task).

from __future__ import print_function
import itertools
import tensorflow as tf
import numpy as np
import sys
import argparse
import datetime




from tensorflow_helpers import *
from data_generator import generate_examples, pick_task
from information_trackers import MutInfSaver, WeightSaver, compute_entropy_fullvec, get_mut_inf_for_vecs,\
                                flat_mutual_inf, compute_avg_entropy_vec
from helper_functions import get_batches, load_pretrained_embeddings, \
                            get_model_type_str, translate_ids_to_words, \
                            save_results, print_into_log
from graph_init import GRU_attractor

ops = {
        'model_type': "GRU", # OPTIONS: vanilla, LSTM_raw, LSTM_tensorflow, LSTM_attractor
        'hid': 5,
        'in': None, #TBD
        'out': 1,
#         'batch_size':n_examples, #since the sequences are 1-dimensional it's easier to just run them all at once
        'n_attractor_iterations': 10,
        'attractor_dynamics': "projection2", # OPTIONS:  "" (for no attractor dynamics),
                                    #           "direct" (simple attractor weights applied to hidden states directly, trained with noise addition)
                                    #           "projection" (project the hidden state into a separate space via weights, do attraction, project back)
                                    #           "helper_hidden" (hidden-hidden neurons) - IMPORTANT: don't forget to add h_hid number
        'h_hid': 10, # helper hidden for "helper hidden" "attractory_dynamics" mode
        'attractor_noise_level': 0.2,
        'attractor_noise_type': "bernoilli", # OPTIONS: "gaussian", "dropout", "random_drop"
    
        'training_mode': "",#'attractor_on_both',
    
        'attractor_regularization': "l2", # OPTIONS: "l2", ""
        'attractor_regularization_lambda': 0.05,
    
        'record_mutual_information': True,
        'problem_type': "parity_length", # OPTIONS: parity, parity_length, majority, reber, kazakov, pos_brown
        'seq_len': 5,
    
        'save_best_model': True, 
        'reshuffle_data_each_replication': False, #relevant for POS datasets (since they are loaded from files)
        'test_partition': 0.3,
    
        # NLP related (pos_brown task)
        'embedding_size': 100,
        'load_word_embeddings': True,
        'train_word_embeddings': True, 
        'input_type': "embed" # embed&prior, embed, prior
        }



# !!!!!!!!!!!!!!!!!!!!!!            
# SEQ_LEN = 12 # number of bits in input sequence   
N_HIDDEN = ops['hid']  # number of hidden units
N_H_HIDDEN = ops['h_hid']          
TASK = ops['problem_type']
ARCH = ops['model_type'] # hidden layer type: 'GRU' or 'tanh'
NOISE_LEVEL = ops['attractor_noise_level']
                      # noise in training attractor net 
                      # if >=0, Gaussian with std dev NOISE_LEVEL 
                      # if < 0, Bernoulli dropout proportion -NOISE_LEVEL 
            
# !!!!!!!!!!!!!!!!!!!!!! 
INPUT_NOISE_LEVEL = 0.1
ATTRACTOR_TYPE = ops['attractor_dynamics']
N_ATTRACTOR_STEPS = ops['n_attractor_iterations']
    #                   # number of time steps in attractor dynamics
    #                   # if = 0, then no attractor net

# !!!!!!!!!!!!!!!!!!!!!!            
# ATTR_WEIGHT_CONSTRAINTS = True
                      # True: make attractor weights symmetric and have zero diag
                      # False: unconstrained
TRAIN_ATTR_WEIGHTS_ON_PREDICTION = False
                      # True: train attractor weights on attractor net _and_ prediction
REPORT_BEST_TRAIN_PERFORMANCE = True
                      # True: save the train/test perf on the epoch for which train perf was best
LOSS_SWITCH_FREQ = 1
                      # how often (in epochs) to switch between attractor 
                      # and prediction loss

ops, SEQ_LEN, N_INPUT, N_CLASSES, N_TRAIN, N_TEST = pick_task(ops['problem_type'], ops)# task (parity, majority, reber, kazakov)


# Training Parameters

TRAINING_EPOCHS = 5000
N_REPLICATIONS = 100
BATCH_SIZE = N_TRAIN
DISPLAY_EPOCH = 500
LRATE_PREDICTION = 0.008
LRATE_ATTRACTOR = 0.008
LOG_DIRECTORY = 'experiments/logs/{}(h_hid).txt'.format(ops['problem_type'])



# NOTEBOOK CODE

WS = WeightSaver()
MIS = MutInfSaver()



######### MAIN CODE #############################################################
for h_hid in [3, 5, 8, 10, 15]:
    # the tf seed needs to be within the context of the graph. 
    tf.reset_default_graph()
    np.random.seed(100)
    tf.set_random_seed(100)
    # ops['n_attractor_iterations'] = attractor_steps
    
    ops['h_hid'] = h_hid
    N_H_HIDDEN = ops['h_hid']   

    #
    # PLACEHOLDERS
    #
    if 'pos' in ops['problem_type']:
        # X will be looked up in the embedding table, so the last dimension is just a number
        X = tf.placeholder("int64", [None, SEQ_LEN], name='X')
        # last dimension is left singular, tensorflow will expect it to be an id number, not 1-hot embed
        Y = tf.placeholder("int64", [None, SEQ_LEN], name='Y')
    else: #single output
        X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
        Y = tf.placeholder("float", [None, N_CLASSES])
    attractor_tgt_net = tf.placeholder("float", [None, N_HIDDEN], name='attractor_tgt')

    # Embedding matrix initialization
    if 'pos' in ops['problem_type']:
        [X_train, Y_train, X_test, Y_test, maps] = generate_examples(SEQ_LEN, N_TRAIN, N_TEST, 
                                                                             INPUT_NOISE_LEVEL, TASK, ops)
        if ops['load_word_embeddings']:
            embeddings_loaded = load_pretrained_embeddings('data/glove.6B.100d.txt', maps, ops)
            embedding = tf.get_variable("embedding",
                                    initializer=embeddings_loaded,
                                    dtype=tf.float32,
                                    trainable=ops['train_word_embeddings'])
        else: #initialize randomly
            embedding = tf.get_variable("embedding",
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    shape=[ops['vocab_size'], ops['embedding_size']],
                                    dtype=tf.float32,
                                    trainable=ops['train_word_embeddings'])
        embed_lookup = tf.nn.embedding_lookup(embedding, X)

        # load priors information
        if ops['input_type'] == 'prior' or ops['input_type'] == 'embed&prior':
            id2prior = maps['id2prior']
            word2id = maps['word2id']
            priors = np.zeros([len(id2prior), len(id2prior[0])]).astype("float32")
            for id, prior in id2prior.items():
                priors[id] = prior
            priors_op = tf.get_variable("priors",
                                    initializer=priors,
                                    dtype=tf.float32,
                                    trainable=False)
            prior_lookup = tf.nn.embedding_lookup(priors_op, X)

        if ops['input_type'] == 'embed':
            embed = embed_lookup
        elif ops['input_type'] == 'prior':
            embed = prior_lookup
        elif ops['input_type'] == 'embed&prior':
            embed = tf.concat([embed_lookup, prior_lookup], axis=2)




    # Graph + all the training variables
    if 'pos' in ops['problem_type']:
        G = GRU_attractor(ops, inputs={'X': embed, 'Y': Y, 'attractor_tgt_net': attractor_tgt_net})
    else:
        G = GRU_attractor(ops, inputs={'X': X, 'Y': Y, 'attractor_tgt_net': attractor_tgt_net})
    pred_loss_op = G.pred_loss_op
    pred_train_op = G.pred_train_op
    attr_loss_op = G.attr_loss_op
    attr_train_op = G.attr_train_op
    accuracy = G.accuracy
    ouptut = G.output

    h_clean_seq = G.h_clean_seq
    h_net_seq = G.h_net_seq
    h_attractor_collection = G.h_attractor_collection



    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    print_into_log(LOG_DIRECTORY, get_model_type_str(ops, N_TRAIN, N_TEST, SEQ_LEN))

    with tf.Session() as sess:
        saved_train_acc = []
        saved_test_acc = []
        saved_epoch = []
        saved_att_loss = []
        saved_entropy_final = []
        # Start training
        for replication in range(N_REPLICATIONS):
            print_into_log(LOG_DIRECTORY, "********** replication " + str(replication) +" **********")
            [X_train, Y_train, X_test, Y_test, maps] = generate_examples(SEQ_LEN, N_TRAIN, N_TEST, 
                                                                         INPUT_NOISE_LEVEL, TASK, ops)

            print(X_test[0:1], Y_test[0:1])
            sess.run(init) # Run the initializer

            train_prediction_loss = True
            best_train_acc = -1000.
            best_test_acc = 0
            best_att_loss = 0
            for epoch in range(1, TRAINING_EPOCHS + 2):
                if (epoch-1) % DISPLAY_EPOCH == 0:
    #                 ploss, train_acc, hid_vals = sess.run([pred_loss_op, accuracy, h_net_seq],
    #                                              feed_dict={X: X_train, Y: Y_train})
                    ploss, train_acc, hid_vals = batch_tensor_collect(sess, [pred_loss_op, accuracy, h_net_seq], 
                                                                      X, Y, X_train, Y_train, BATCH_SIZE)
                    aloss = []
                    for batch_hid_vals in np.array_split(hid_vals, int(len(X_train)/BATCH_SIZE)):
                        aloss.append(sess.run(attr_loss_op,feed_dict={attractor_tgt_net: batch_hid_vals}))
                    aloss = np.mean(aloss)

    #                 test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
                    test_acc = batch_tensor_collect(sess, [accuracy], X, Y, X_test, Y_test, BATCH_SIZE)[0]
                    if ops['record_mutual_information']:
    #                     h_attractor_val, h_clean_val = sess.run([h_attractor_collection, h_clean_seq],
    #                                                                    feed_dict={X: X_train, Y: Y_train})
                        # TODO: h_attractor_collection reshapeing masking. 
                        h_attractor_val = None
                        h_clean_val = batch_tensor_collect(sess, [h_clean_seq],
                                                                            X, Y, X_train, Y_train, BATCH_SIZE)[0]
                        MIS.update(ploss, aloss, train_acc, test_acc, np.tanh(hid_vals), h_attractor_val, h_clean_val)


                    # Print some examples:
                    if 'pos' in ops['problem_type']:
                        random_ids = np.random.choice(ops['vocab_size'], 1, replace=False)
                        Y_pred = sess.run(ouptut, feed_dict={X: X_train[random_ids]})
                        Y_pred = np.argmax(Y_pred, axis=2)
                        print(random_ids)
                        for i, id in enumerate(random_ids):
                            translate_ids_to_words(X_train[id], Y_pred[i], Y_train[id], maps['id2word'], maps['id2tag'],
                                                  printout=True)
                        print('\n')
                    # Print training information
                    progress_comment = "epoch " + str(epoch-1) + ", Loss Pred " + \
                              "{:.4f}".format(ploss) + ", Loss Att " + \
                              "{:.4f}".format(aloss) + ", Train Acc= " + \
                              "{:.3f}".format(train_acc) + ", Test Acc= " + \
                              "{:.4f}".format(test_acc) + ", Entropy= " + \
                              "{:.4f}".format(compute_entropy_fullvec(h_clean_val, ops, n_bins=8))
                    print_into_log(LOG_DIRECTORY, progress_comment)
                        
                    WS.update_conservative(
                                    epoch_number = epoch,
                                    loss_att = aloss, 
                                    loss_task = ploss, 
    #                                 W_att=sess.run(attr_net['W']), 
    #                                 b_att = sess.run(attr_net['W']), 
    #                                 scaling_const = None,
                                    acc = test_acc,
    #                                 h_seq = None
                                    )

                    if (train_acc > best_train_acc):
                        best_train_acc = train_acc
                        best_test_acc = test_acc
                        best_att_loss = aloss
                        # TODO
    #                     save_path = saver.save(sess, "/experiments/models/.ckpt")
                    if (1.0 - train_acc - 1e-15 < 0.0):
                        print('reached_peak')
                        break
                if epoch > 1 and LOSS_SWITCH_FREQ > 0 \
                             and (epoch-1) % LOSS_SWITCH_FREQ == 0:
                    train_prediction_loss = not train_prediction_loss
                batches = get_batches(BATCH_SIZE, X_train, Y_train)
                for (batch_x, batch_y) in batches:
                    if (LOSS_SWITCH_FREQ == 0 or train_prediction_loss):
                        # Optimize all parameters except for attractor weights
                        _, hid_vals = sess.run([pred_train_op, h_net_seq], 
                                               feed_dict={X: batch_x, Y: batch_y})
                    if (LOSS_SWITCH_FREQ == 0 or not train_prediction_loss):
                        if (N_ATTRACTOR_STEPS > 0):
                            # Optimize attractor weights
                            hid_vals = sess.run(h_net_seq, feed_dict={X: batch_x, Y: batch_y})
                            sess.run(attr_train_op, feed_dict={attractor_tgt_net: 
                                                           hid_vals})

            print("Optimization Finished!")

        
            if (REPORT_BEST_TRAIN_PERFORMANCE):
                saved_train_acc.append(best_train_acc)
                saved_test_acc.append(best_test_acc)
                saved_att_loss.append(best_att_loss)
                saved_entropy_final.append(compute_entropy_fullvec(MIS.h_finals[-1], ops, n_bins=8))
            else: 
                saved_train_acc.append(train_acc)
                saved_test_acc.append(test_acc)
    #             saved_att_loss.append(aloss)
            if (train_acc == 1.0):
                saved_epoch.append(epoch)

            # print weights
            #for p in attr_net.values():
            #    print (p.name, ' ', p.eval())
        print('********************************************************************')
        print('********************************************************************')
        print('mean train accuracy', np.mean(saved_train_acc))
        print('indiv runs ',saved_train_acc)
        print('mean epoch', np.mean(saved_epoch))
        print('indiv epochs ',saved_epoch)
        print('mean test accuracy', np.mean(saved_test_acc))
        print('indiv runs ',saved_test_acc)
        save_results(ops, saved_train_acc, saved_test_acc, saved_att_loss, saved_entropy_final,
                     N_TRAIN, N_TEST, SEQ_LEN, comment="h_hid_test")



