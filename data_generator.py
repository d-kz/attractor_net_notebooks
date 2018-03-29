import numpy as np
import itertools
import fsm
import pickle
import random



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
    if (task == 'parity' or task == 'parity_length'):
        X_train, Y_train = generate_parity_majority_sequences(seq_len, n_train, task)
        X_test, Y_test = generate_parity_majority_sequences(seq_len, n_test, task)
        if (input_noise_level > 0.):
           X_test, Y_test = add_input_noise(input_noise_level,X_test,Y_test,2)
    # for majority, split all sequences into training and test sets
    elif (task == 'majority'):
        X_train, Y_train = generate_parity_majority_sequences(seq_len, n_train+n_test, task)
        pix = np.random.permutation(n_train+n_test)
        X_train = X_train[pix[:n_train],:]
        Y_train = Y_train[pix[:n_train],:]
        X_test = X_train[pix[n_train:],:]
        Y_test = Y_train[pix[n_train:],:]
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


    return [X_train, Y_train, X_test, Y_test, maps]

################ generate_parity_majority_sequences #####################################
def generate_parity_majority_sequences(N, count, task):
    """
    Generate :count: sequences of length :N:.
    If odd # of 1's -> output 1
    else -> output 0
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
        dataset_X, dataset_Y = np.array(dataset['X']), np.array(dataset['Y'])

    map_names = ['id2tag', "tag2id", "id2word", "word2id", "id2prior"]
    maps = {map_name: [] for map_name in map_names}
    for map_name in map_names:
        with open(directory + "/" + map_name + '.pickle', 'rb') as handle:
            maps[map_name] = pickle.load(handle)

    return [dataset_X, dataset_Y, maps]



def pick_task(task_name, ops):
    if (task_name=='parity'):
        SEQ_LEN = 5
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = pow(2,SEQ_LEN) # train on all seqs
        N_TEST = pow(2,SEQ_LEN)
        solved_problem_count = 0
    if (task_name=='parity_length'):
        SEQ_LEN = 12
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = 1000#1000 # train on all seqs
        N_TEST = 1000#1000#pow(2,SEQ_LEN)
        solved_problem_count = 0
    elif (task_name=='majority'):
        SEQ_LEN = 5
        N_INPUT = 1           # number of input units
        N_CLASSES = 1         # number of output units
        N_TRAIN = 64 
        N_TEST = 4096-64
    elif (task_name=='reber'):
        SEQ_LEN = 20
        N_INPUT = 7 # B E P S T V X
        N_CLASSES = 1
        N_TRAIN = 200
        N_TEST = 2000
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
    else:
        print('Invalid task: ',task_name)
    ops['in'] = N_INPUT
    ops['out'] = N_CLASSES
    return ops, SEQ_LEN, N_INPUT, N_CLASSES, N_TRAIN, N_TEST