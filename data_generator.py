import numpy as np
import itertools
import fsm
import pickle
import random
import gzip
import sys
import json
if (sys.version_info > (3, 0)):
    import pickle as pkl
else:
    #Python 2.7 imports
    import cPickle as pkl




################ add_input_noise ########################################################
# incorporate input noise into the test patterns
def add_input_noise(noise_level, X, Y, n_repeat):
# X: # examples X # sequence elements X #inputs
    X = np.repeat(X, n_repeat, axis=0)
    Y = np.repeat(Y, n_repeat, axis=0)
    X = X + (np.random.random(X.shape)*2.0-1.0) * noise_level
    return X,Y


################ generate_examples ######################################################
def generate_examples(seq_len, n_train, n_test, input_noise_level, task, ops):
    cutoff_seq = lambda x, len: x[0:len]
    maps = None
    X_val, Y_val = None, None
    print(task)
    if (task == 'parity' or task == 'parity_length'):
        X_train, Y_train = generate_parity_majority_sequences(seq_len, n_train, task)
        X_test, Y_test = generate_parity_majority_sequences(seq_len, n_test, task)
        if (input_noise_level > 0.):
           X_test, Y_test = add_input_noise(input_noise_level,X_test,Y_test,2)
    # for majority, split all sequences into training and test sets
    elif task == 'parity_length_noisy_longer_remainder':
        X, Y = generate_parity_majority_sequences(seq_len, n_train, task)
        # remainder
        train_split = int((n_train * 0.25))
        X_train, Y_train, X_test_left, Y_test_left = X[0:train_split], Y[0:train_split], X[train_split:], Y[train_split:]
        # noisy
        if (input_noise_level > 0.):
            X_test_noise, Y_test_noise = add_input_noise(input_noise_level*2, X_train, Y_train, 2)
        # longer
        longer_len = seq_len*3
        X_test_long, Y_test_long = generate_parity_majority_sequences(longer_len, len(X), task)

        X_test = [X_test_left, X_test_noise, X_test_long]
        Y_test = [Y_test_left, Y_test_noise, Y_test_long]
    elif (task == 'majority'):
        X, Y = generate_parity_majority_sequences(seq_len, n_train+n_test, task)
        pix = np.random.permutation(n_train+n_test)
        X_train = X[pix[:n_train],:]
        Y_train = Y[pix[:n_train],:]
        X_test = X[pix[n_train:],:]
        Y_test = Y[pix[n_train:],:]
        if (input_noise_level > 0.):
           X_test, Y_test = add_input_noise(input_noise_level,X_test,Y_test,1)
    elif (task == 'reber'):
        _, Y_train, X_train, _ = fsm.generate_grammar_dataset(1, seq_len, n_train)
        _, Y_test, X_test, _ = fsm.generate_grammar_dataset(1, seq_len, n_test)
    elif (task == 'kazakov'):
        _, Y_train, X_train, _ = fsm.generate_grammar_dataset(2, seq_len, n_train)
        _, Y_test, X_test, _ = fsm.generate_grammar_dataset(2, seq_len, n_test)
    elif (task == "pos_brown"):
        dataset_X, dataset_Y, maps = get_pos_brown_dataset('data/corpus_brown', ops['test_partition'])
        # trip sequences to seq_len
        # dataset_X = np.array([cutoff_seq(x) for x in dataset_X], ops['seq_len'])
        # dataset_Y = np.array([cutoff_seq(x) for x in dataset_Y], ops['seq_len'])
        if ops['reshuffle_data_each_replication']:
            indeces = range(dataset_X.shape[0])
            random.shuffle(indeces)
            dataset_X = dataset_X[indeces]
            dataset_Y = dataset_Y[indeces]
        max_len = np.max([len(x) for x in dataset_X])
        # pad all sequences with zeros at the end
        X = np.zeros([len(dataset_X), max_len])
        Y = np.zeros([len(dataset_X), max_len])
        for i,x in enumerate(dataset_X):
            X[i,:len(x)] = x
        for i,x in enumerate(dataset_Y):
            Y[i,:len(x)] = x
        X = X.astype("int64")
        Y = Y.astype("int64")

        X_test = X[0:n_test,:]
        print(X_test.shape)
        Y_test = Y[0:n_test,:]
        X_train = X[n_test:,:]
        Y_train = Y[n_test:,:]

        val_cut = int(0.2*X_train.shape[0])
        X_val = X_train[0:val_cut,:]
        Y_val = Y_train[0:val_cut,:]
        X_train = X_train[val_cut:,:]
        Y_train = Y_train[val_cut:,:]

    elif (task == "ner_german"):
        X, Y, maps, embeddings_dict = get_ner_german_dataset('data/ner_german')
        # 16% test, 7% dev
        test_cut = int(len(Y)*0.17)
        val_cut = int(len(Y) * 0.07)
        train_cut = int(len(Y) - (test_cut + val_cut))
        # train_data, test_data, dev_data (like was in the original dataset)
        X_train = X[0:train_cut, :, :]
        Y_train = Y[0:train_cut, :]

        X_test = X[train_cut:train_cut + test_cut, :, :]
        Y_test = Y[train_cut:train_cut + test_cut, :]

        X_val = X[train_cut + test_cut:, :, :]
        Y_val = Y[train_cut + test_cut:, :]

    elif (task == "sentiment_imdb"):
        X, Y, maps = get_sentiment_imbd('data/imdb_keras')

        Y = np.expand_dims(Y, axis=1)
        # 16% test, 7% dev
        test_cut = int(len(Y)*0.20)
        val_cut = int(len(Y) * 0.20)
        train_cut = int(len(Y) - (test_cut + val_cut))
        # train_data, test_data, dev_data (like was in the original dataset)
        print(X.shape, Y.shape)
        X_train = X[0:train_cut, :]
        Y_train = Y[0:train_cut, :]

        X_test = X[train_cut:train_cut + test_cut, :]
        Y_test = Y[train_cut:train_cut + test_cut, :]

        X_val = X[train_cut + test_cut:, :]
        Y_val = Y[train_cut + test_cut:, :]

    elif (task == "topic_classification"):
        X, Y, maps = get_topic_classification_reuters('data/topic_classification')

        Y = np.expand_dims(Y, axis=1)
        # 16% test, 7% dev
        test_cut = int(len(Y)*0.30)
        train_cut = int(len(Y) - (test_cut))
        # train_data, test_data, dev_data (like was in the original dataset)
        print(X.shape, Y.shape)
        X_train = X[0:train_cut, :]
        Y_train = Y[0:train_cut, :]

        X_test = X[train_cut:, :]
        Y_test = Y[train_cut:, :]
    elif (task == "video_classification"):
        with open('data/video_classification/data_class25.pickle', 'rb') as handle:
            dataset = json.load(handle)
            X_train, Y_train, X_test, Y_test = np.array(dataset['X_train']), np.array(dataset['Y_train']), np.array(dataset['X_test']), np.array(dataset['Y_test'])
        Y_test = np.expand_dims(Y_test, axis=1)
        Y_train = np.expand_dims(Y_train, axis=1)
    elif (task == "msnbc"):
        with open('data/msnbc/data.pickle', 'rb') as handle:
            dataset = pickle.load(handle)
            X_train, Y_train, X_test, Y_test = np.array(dataset['X_train']), np.array(dataset['Y_train']), np.array(dataset['X_test']), np.array(dataset['Y_test'])

        X_train_pad = np.zeros([len(X_train), 40])
        Y_train_pad = np.zeros([len(X_train), 40])
        for i, x in enumerate(X_train):
            X_train_pad[i, :len(x)] = x
        for i, x in enumerate(Y_train):
            Y_train_pad[i, :len(x)] = x
        X_train = X_train_pad.astype("int64")
        Y_train = Y_train_pad.astype("int64")

        X_test_pad = np.zeros([len(X_test), 40])
        Y_test_pad = np.zeros([len(X_test), 40])
        for i, x in enumerate(X_test):
            X_test_pad[i, :len(x)] = x
        for i, x in enumerate(Y_test):
            Y_test_pad[i, :len(x)] = x
        X_test = X_test_pad.astype("int64")
        Y_test = Y_test_pad.astype("int64")

    return [X_train, Y_train, X_test, Y_test, X_val, Y_val, maps]

################ generate_parity_majority_sequences #####################################
def generate_parity_majority_sequences(N, count, task):
    """
    Generate :count: sequences of length :N:.
    If odd # of 1's -> output 1
    else -> output 0
    If count >= 2**N (possible sequences exceeded), then generate the dataset with permutation.
    """
    parity = lambda x: 1 if (x % 2 == 1) else 0
    majority = lambda x: 1 if x > N/2 else 0
    if (count >= 2**N):
        sequences = np.asarray([seq for seq in itertools.product([0,1],repeat=N)])
    else:
        sequences = np.random.choice([0, 1], size=[count, N], replace=True)
    counts = np.count_nonzero(sequences == 1, axis=1)
    # parity each sequence, expand dimensions by 1 to match sequences shape
    if (task == 'parity' or task == 'parity_length'):
        y = np.expand_dims(np.array([parity(x) for x in counts]), axis=1)
    else: # majority
        y = np.expand_dims(np.array([majority(x) for x in counts]), axis=1)

    # In case if you wanted to have the answer just appended at the end of the sequence:
    #     # append the answer at the end of each sequence
    #     seq_plus_y = np.concatenate([sequences, y], axis=1)
    #     print(sequences.shape, y.shape, seq_plus_y.shape)
    #     return seq_plus_y
    return np.expand_dims(sequences, axis=2), y

def get_pos_brown_dataset(directory, partition):
    with open(directory + "/data.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        dataset_X, dataset_Y = dataset['X'], dataset['Y']

    map_names = ['id2tag', "tag2id", "id2word", "word2id", "id2prior"]
    maps = {map_name: [] for map_name in map_names}
    for map_name in map_names:
        with open(directory + "/" + map_name + '.pickle', 'rb') as handle:
            maps[map_name] = pickle.load(handle)

    return [dataset_X, dataset_Y, maps]

def get_sentiment_imbd(directory):
    with open(directory + "/dataset.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        dataset_X, dataset_Y = np.array(dataset['X']), np.array(dataset['Y'])

    map_names = ["id2word", "word2id"]
    maps = {}
    for map_name in map_names:
        with open(directory + '/maps.pickle', 'rb') as handle:
            maps = pickle.load(handle)
    with open("data/imdb_keras/dataset_params.pickle", 'rb') as handle:
        dataset_params = pickle.load(handle)
        SEQ_LEN = dataset_params['seq_len_max']

    X = np.zeros([len(dataset_X), SEQ_LEN])
    Y = np.zeros([len(dataset_X)])
    for i, x in enumerate(dataset_X):
        X[i, :len(x)] = x
    for i, x in enumerate(dataset_Y):
        Y[i] = x
    X = X.astype("int64")
    Y = Y.astype("int64")

    return [X, Y, maps]

def get_topic_classification_reuters(directory):
    with open(directory + "/dataset.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        dataset_X, dataset_Y = np.array(dataset['X']), np.array(dataset['Y'])

    map_names = ["id2word", "word2id"]
    maps = {}
    for map_name in map_names:
        with open(directory + '/maps.pickle', 'rb') as handle:
            maps = pickle.load(handle)
    with open(directory + "/dataset_params.pickle", 'rb') as handle:
        dataset_params = pickle.load(handle)
        SEQ_LEN = dataset_params['seq_len_max']

    X = np.zeros([len(dataset_X), SEQ_LEN])
    Y = np.zeros([len(dataset_X)])
    for i, x in enumerate(dataset_X):
        X[i, :len(x)] = x
    for i, x in enumerate(dataset_Y):
        Y[i] = x
    X = X.astype("int64")
    Y = Y.astype("int64")

    return [X, Y, maps]

def get_ner_german_dataset(directory):
    with open(directory + "/dataset_cutoff.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
        tokens, casing, labels  = dataset['tokens'], dataset['casing'], dataset['Y']
    with open("data/ner_german/data_params.pickle", 'rb') as handle:
        dataset_params = pickle.load(handle)

    # embeddinigs:
    f = gzip.open('data/ner_german/embeddings.pkl.gz', 'rb')
    embeddings = pkl.load(f)
    f.close()

    label2Idx = embeddings['label2Idx']
    wordEmbeddings = embeddings['wordEmbeddings']
    caseEmbeddings = embeddings['caseEmbeddings']
    # Inverse label mapping
    idx2Label = {v: k for k, v in label2Idx.items()}
    maps = {'id2tag': idx2Label, 'tag2id': label2Idx}
    embeddings_dict = {'word': wordEmbeddings, "case": caseEmbeddings}

    # putting the embeddings in
    X_tok = np.zeros([len(tokens), dataset_params['seq_len_max'], len(wordEmbeddings[0])])
    X_cas = np.zeros([len(tokens), dataset_params['seq_len_max'], len(caseEmbeddings[0])])
    Y = np.zeros([len(tokens), dataset_params['seq_len_max']])
    for i, x in enumerate(tokens):
        X_tok[i, :len(x), :] = wordEmbeddings[x]
    for i, x in enumerate(casing):
        X_cas[i, :len(x), :] = caseEmbeddings[x]
    for i, x in enumerate(labels):
        Y[i, :len(x)] = x

    X = np.concatenate([X_tok, X_cas], axis=2)
    # putting in all the embeddings right away
    return [X, Y, maps, embeddings_dict]



def pick_task(task_name, ops):
    if (task_name=='parity'):
        SEQ_LEN = 10
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = 256 # train on all seqs
        N_TEST = 768
        solved_problem_count = 0
    elif (task_name=='parity_length'):
        SEQ_LEN = 12
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = 1000#1000 # train on all seqs
        N_TEST = 1000#1000#pow(2,SEQ_LEN)
        solved_problem_count = 0
    elif (task_name=='parity_length_noisy_longer_remainder'):
        SEQ_LEN = 10
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = pow(2,SEQ_LEN) #1000 # train on all seqs
        N_TEST = pow(2,SEQ_LEN)#1000#pow(2,SEQ_LEN)
    elif (task_name=='majority'):
        SEQ_LEN = 10
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = 256
        N_TEST = 768
    elif (task_name=='reber'):
        SEQ_LEN = 20
        N_INPUT = 7 # B E P S T V X
        N_CLASSES = 1
        N_TRAIN = 200
        N_TEST = 400
    elif (task_name=='kazakov'):
        SEQ_LEN = 20
        N_INPUT = 5
        N_CLASSES = 1
        N_TRAIN = 400
        N_TEST = 2000
    elif(task_name=='pos_brown'):
        with open("data/corpus_brown/data_params.pickle", 'rb') as handle:
            dataset_params = pickle.load(handle)
        SEQ_LEN = dataset_params['seq_len_max'] #length 50 cutoff preserves 98% of data
        if ops['input_type'] == 'embed':
            N_INPUT = ops['embedding_size'] # embedding vector size
        elif ops['input_type'] == 'prior':
            N_INPUT = 147
        elif ops['input_type'] == 'embed&prior':
            N_INPUT = 147 + ops['embedding_size']

        N_CLASSES = dataset_params['n_classes'] # tags size
        total_examples = dataset_params['total_examples'] # total sequences
        N_TEST = int(total_examples*ops['test_partition'])
        N_TRAIN = total_examples - N_TEST
        ops['seq_len'] = SEQ_LEN
        ops['vocab_size'] = dataset_params['vocab_size']
    elif (task_name == 'ner_german'):
        N_INPUT = 108 # word embed + case embed
        with open("data/ner_german/data_params.pickle", 'rb') as handle:
            dataset_params = pickle.load(handle)
        SEQ_LEN = dataset_params['seq_len_max']
        N_CLASSES = dataset_params['n_classes']
        total = dataset_params['total_examples']
        N_TEST = int(total * 0.17)
        N_VALID = int(total * 0.07)
        N_TRAIN = int(total - (N_TEST + N_VALID))
        ops['seq_len'] = SEQ_LEN
    elif (task_name == 'sentiment_imdb'):
        N_INPUT = ops['embedding_size'] # word embed
        with open("data/imdb_keras/dataset_params.pickle", 'rb') as handle:
            dataset_params = pickle.load(handle)
        SEQ_LEN = dataset_params['seq_len_max']
        N_CLASSES = 1 # output is singular since only 2 classes.
        total = dataset_params['total_examples']
        N_TEST = int(total * 0.20)
        N_VALID = int(total * 0.20)
        N_TRAIN = int(total - (N_TEST + N_VALID))
        ops['seq_len'] = SEQ_LEN
    elif (task_name == "topic_classification"):
        N_INPUT = ops['embedding_size']  # word embed
        with open("data/topic_classification/dataset_params.pickle", 'rb') as handle:
            dataset_params = pickle.load(handle)
        SEQ_LEN = dataset_params['seq_len_max']
        N_CLASSES = dataset_params['n_classes']  # output is singular since only 2 classes.
        total = dataset_params['total_examples']
        N_TEST = int(total * 0.20)
        N_VALID = int(total * 0.20)
        N_TRAIN = int(total - (N_TEST + N_VALID))
        ops['seq_len'] = SEQ_LEN
    elif (task_name == "video_classification"):
        N_INPUT = 2048 # word embed
        SEQ_LEN = 40
        N_CLASSES = 25  # output is singular since only 2 classes.
        N_TEST = 0
        N_VALID = 0
        N_TRAIN = 0
        ops['seq_len'] = SEQ_LEN
    elif (task_name == "msnbc"):
        N_INPUT = 18 # word embed
        SEQ_LEN = 40
        N_CLASSES = 18  # output is singular since only 2 classes.
        N_TEST = 0
        N_VALID = 0
        N_TRAIN = 0
        ops['seq_len'] = SEQ_LEN
    else:
        print('Invalid task: ',task_name)
    ops['in'] = N_INPUT
    ops['out'] = N_CLASSES
    return ops, SEQ_LEN, N_INPUT, N_CLASSES, N_TRAIN, N_TEST