import numpy as np
import datetime



def load_pretrained_embeddings(embedding_path, maps, ops):
    # load embeddigns
    print("Loading embeddings...")
    f = open(embedding_path, 'r')
    pretrained_emb = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        pretrained_emb[word] = embedding
    print(len(pretrained_emb), " words loaded!")

    # align embeddings to dictionary used in dataset:
    words_not_found = []
    # TODO: not sure what's the best way to handle this since some unicode words don't get processed, but we don't
    # want to mess up the id2word tagging
    unique_words_in_data = np.max(maps['id2word'].keys()) + 1
    print(unique_words_in_data)
    embeddings = np.random.uniform(low=-1.0, high=1.0, size=[unique_words_in_data, ops['embedding_size']]).astype("float32")
    for id, word in maps['id2word'].items():
        if word in pretrained_emb:
            embeddings[id] = pretrained_emb[word]
        else:
            words_not_found.append(word)

    print("{} words not found in pretrained embeddings: {}".format(len(words_not_found), words_not_found))
    return embeddings


################ get_batches ############################################################
def get_batches(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    batches = []
    while len(idx) > 0:
       cur_idx = idx[:min(num, len(idx))]

       data_shuffle = [data[i] for i in cur_idx]
       labels_shuffle = [labels[i] for i in cur_idx]
       batches.append((np.asarray(data_shuffle), np.asarray(labels_shuffle)))
       idx = idx[num:]
    return batches

# def get_model_title():


def get_model_type_str(ops, N_TRAIN, N_TEST, SEQ_LEN):
    comment = """
    model_type: \t\t{} bidir({}), task: {}
    hid: \t\t\t{},
    h_hid: \t\t\t{}
    n_attractor_iterations: \t{},
    attractor_dynamics: \t{}
    attractor_noise_level: \t{}
    attractor_noise_type: \t{}
    attractor_regu-n: \t\t{}(lambda:{})
    word_embedding: size\t({}), train({})
    dropout: \t\t\t{}
    TRAIN/TEST_SIZE: \t{}/{}, SEQ_LEN: {}""".format(ops['model_type'], ops['bidirectional'], ops['problem_type'], ops['hid'], ops['h_hid'],
                                                    ops['n_attractor_iterations'],
                                                    ops['attractor_dynamics'], ops['attractor_noise_level'],
                                                    ops['attractor_noise_type'],
                                                    ops['attractor_regularization'],
                                                    ops['attractor_regularization_lambda'],
                                                    ops['embedding_size'], ops['train_word_embeddings'],
                                                    ops['dropout'],
                                                    N_TRAIN, N_TEST, SEQ_LEN)

    return comment

def get_training_progress_comment(epoch, ploss, aloss, ploss_val, val_acc, train_acc, test_acc, entropy):
    progress_comment = "epoch=" + str(epoch - 1) + "; Loss Pred=" + \
                       "{:.4f}".format(ploss) + \
                       "; Val Loss={:.4f}".format(ploss_val) + \
                       "; Val Acc={:.4f}".format(val_acc) + \
                       "; Loss Att=" + \
                       "{}".format(aloss) + "; Train Acc=" + \
                       "{:.3f}".format(train_acc) + "; Test Acc=" + \
                       "{:.4f}".format(test_acc) + "; Entropy=" + \
                       "{}".format(entropy) + "\n"
    return progress_comment

#################
def print_some_translated_sentences(sess, output, X, X_train, Y_train, maps, ops, n_rand_sentences=1):
    """
    Only relevant for NLP tasks
    """""
    random_ids = np.random.choice(ops['vocab_size'], n_rand_sentences, replace=False)
    Y_pred = sess.run(output, feed_dict={X: X_train[random_ids]})
    Y_pred = np.argmax(Y_pred, axis=2)
    print(random_ids)
    for i, id in enumerate(random_ids):
        translate_ids_to_words(X_train[id], Y_pred[:,i], Y_train[id], maps['id2word'], maps['id2tag'],
                               printout=True)
    print('\n')


def translate_ids_to_words(x, y, y_true, id2word, id2tag, printout=False, log=False):
    # prints translated sequences and logs them.
    words = [id2word[id] for id in x]
    tags = [id2tag[id] for id in y]
    true_tags = [id2tag[id] for id in y_true]
    print_str = ''
    for i in range(len(x)):
        if words[i] != 'PAD':
            if y[i] == y_true[i]: #print in green
                print_str += "{}<\033[1;32m{}\x1b[0m> ".format(words[i], tags[i])
            else: #print in red (green for correct answer)
                ""
                print_str += "{}<\x1b[31m{}\x1b[0m(\033[1;32m{}\x1b[0m)> ".format(words[i], tags[i], true_tags[i])
    if printout:
        print(print_str)
    # TODO: logging
    return print_str



##################
def save_results(ops, saved_epochs, saved_train_acc, saved_test_acc, saved_att_loss, saved_entropy_final, saved_val_acc,
                 saved_val_loss, saved_train_loss, N_TRAIN, N_TEST, SEQ_LEN, comment):
    saved_train_acc = np.array(saved_train_acc)
    saved_test_acc = np.array(saved_test_acc)
    saved_att_loss = np.array(saved_att_loss)
    saved_entropy_final = np.array(saved_entropy_final)
    saved_epochs = np.array(saved_epochs)
    saved_val_acc = np.array(saved_val_acc)
    saved_val_loss = np.array(saved_val_loss)
    saved_train_loss = np.array(saved_train_loss)

    np.set_printoptions(precision=3)
    try:
        results = "\n<RESULTS>:\ntype: \t\t\tmean: \t var: \t\n"
        results += "{} \t{:.4f} \t {:.4f}\n".format("saved_train_acc", np.mean(saved_train_acc),
                                                    np.var(saved_train_acc))
        results += "{} \t\t{:.4f} \t {:.4f}\n".format("saved_test_acc", np.mean(saved_test_acc), np.var(saved_test_acc))
        results += "{} \t\t{:.4f} \t {:.4f}\n".format("saved_att_loss", np.mean(saved_att_loss), np.var(saved_att_loss))
        results += "{} \t\t{:.4f} \t {:.4f}\n".format("saved_entropy_final", np.mean(saved_entropy_final), np.var(saved_entropy_final))

        results += "\nEPOCHS:" + np.array2string(saved_epochs, formatter={'float_kind': lambda x: "%.3f" % x})
        results += "\nTRAIN:" + np.array2string(saved_train_acc, formatter={'float_kind': lambda x: "%.3f" % x})
        results += "\nTEST:" + np.array2string(saved_test_acc, formatter={'float_kind': lambda x: "%.3f" % x})
        results += "\nATT_LOSS:" + np.array2string(saved_att_loss, formatter={'float_kind': lambda x: "%.3f" % x})
        results += "\nENTROPY:" + np.array2string(saved_entropy_final, formatter={'float_kind': lambda x: "%.3f" % x})

        results += "\nsaved_val_acc:" + np.array2string(saved_val_acc , formatter={'float_kind': lambda x: "%.3f" % x})
        results += "\nsaved_val_loss:" + np.array2string(saved_val_loss , formatter={'float_kind': lambda x: "%.3f" % x})
        results += "\nsaved_train_loss:" + np.array2string(saved_train_loss , formatter={'float_kind': lambda x: "%.3f" % x})




    except: # if failed to convert to numbers, just print all as string for now TODO
        results += "\nEPOCHS:" + ';'.join([str(el) for el in saved_epochs])
        results += "\nTRAIN:" + ';'.join([str(el) for el in saved_train_acc])
        results += "\nTEST:" + ';'.join([str(el) for el in saved_test_acc])
        results += "\nATT_LOSS:" + ';'.join([str(el) for el in saved_att_loss])
        results += "\nENTROPY:" + ';'.join([str(el) for el in saved_entropy_final])

        results += "\nsaved_val_acc:" + ';'.join([str(el) for el in saved_val_acc])
        results += "\nsaved_val_loss:" + ';'.join([str(el) for el in saved_val_loss])
        results += "\nsaved_train_loss:" + ';'.join([str(el) for el in saved_train_loss])

    #         results += "{} \t {:.4f} \t {:.4f}\n".format("saved_att_loss", np.mean(saved_att_loss), np.var(saved_att_loss))


    title = "../attractor_net_notebooks/experiments/results/{}.txt".format(ops['problem_type'])
    text = """\n
({}): {}
<NETWORK>:
model_type: \t\t{},
hid: \t\t\t{},
h_hid: \t\t\t{}
n_attractor_iterations: {},
attractor_dynamics: \t{}
attractor_noise_level: \t{}
attractor_noise_type: \t{}
attractor_regu-n: \t{}(lambda:{})
TRAIN/TEST_SIZE: \t{}/{}, SEQ_LEN: {}""".format(datetime.date.today(), comment, ops['model_type'], ops['hid'],
                                            ops['h_hid'], ops['n_attractor_iterations'],
                                            ops['attractor_dynamics'], ops['attractor_noise_level'],
                                            ops['attractor_noise_type'],
                                            ops['attractor_regularization'], ops['attractor_regularization_lambda'],
                                            N_TRAIN, N_TEST, SEQ_LEN)

    text += results
    with open(title, "a") as myfile:
        myfile.write(text)
        print("Saved Results Successfully")

def print_into_log(log_dir, comment='', supress=False):
    with open(log_dir, "a") as myfile:
        myfile.write(comment)
        print("Logged Successfully: ")
        if not supress:
            print(comment)