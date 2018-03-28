import numpy as np


def load_pretrained_embeddings(embedding_path, maps, N_INPUT):
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
    unique_words_in_data = len(maps['id2word'])

    embeddings = np.random.uniform(low=-1.0, high=1.0, size=[unique_words_in_data, N_INPUT]).astype("float32")
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


def print_model_type(ops, N_TRAIN, N_TEST, SEQ_LEN):
    print("""
    model_type: \t\t{}, task: {}
    hid: \t\t\t{},
    h_hid: \t\t\t{}
    n_attractor_iterations: \t{},
    attractor_dynamics: \t{}
    attractor_noise_level: \t{}
    attractor_noise_type: \t{}
    attractor_regu-n: \t{}(lambda:{})
    train_word_embeddings: \t{}
    TRAIN/TEST_SIZE: \t{}/{}, SEQ_LEN: {}""".format(ops['model_type'], ops['problem_type'], ops['hid'], ops['h_hid'],
                                                    ops['n_attractor_iterations'],
                                                    ops['attractor_dynamics'], ops['attractor_noise_level'],
                                                    ops['attractor_noise_type'],
                                                    ops['attractor_regularization'],
                                                    ops['attractor_regularization_lambda'],
                                                    ops['train_word_embeddings'],
                                                    N_TRAIN, N_TEST, SEQ_LEN))




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
