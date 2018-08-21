import numpy as np

def generate_single_example(l, n_filler, s, errtype):
    # errtype = 0 for symmetric
    # errtype = 1 for symbol substitution
    # errtype = 2 for symbol swap
    # l: length (must be odd)

    rndseqlen = (l - n_filler) / 2
    while True:
        x = [np.random.randint(1,s) for i in range(rndseqlen)]
        xrev = x[::-1]
        x = x + xrev

        exchangeable = []
        for posn in range(len(x)-1):
            if (x[posn] != x[posn+1]):
               exchangeable += [posn]
        if (len(exchangeable) > 0):
           break

    if errtype == 1: # exchange two symbols
        # make a list of all adjacent symbols that are different

        ix = np.random.randint(0,len(exchangeable))
        posn = exchangeable[ix]
        x[posn], x[posn+1] = x[posn+1], x[posn]

    if errtype == 2: # substitute a symbol
        posn = np.random.randint(0,len(x))
        while True:
            r = np.random.randint(1,s)
            if (r != x[posn]):
                break
        x[posn] = r

    for i in range(n_filler):
        x.insert(rndseqlen, 0) # insert filler symbol

    return x

def embed_one_hot(batch_array, depth):
    """
    :batch_array: gets shaped into one-hot vectors of :depth:
    * :batch_array: needs to start being indexed from 0 for its elements.
    """
    batch_size, length = len(batch_array), len(batch_array[0])
    if batch_size == 0.0:
        batch_size = batch_array.shape[0]

    one_hot_matrix = np.zeros((batch_size, length, depth))
    for i,array in enumerate(batch_array):
        one_hot_matrix[i, np.arange(len(array)), array] = 1
    return one_hot_matrix


def generate_symmetry_dataset(seq_len, n_filler, n_sym, n_examples):

    x = [generate_single_example(seq_len, n_filler, n_sym, 0) for i in range(n_examples/2)]
    y = [1 for i in range(n_examples/2)]
    x += [generate_single_example(seq_len, n_filler, n_sym, 1) for i in range(n_examples/4)]
    x += [generate_single_example(seq_len, n_filler, n_sym, 2) for i in range(n_examples/4)]
    y += [0 for i in range(n_examples/2)]

    # convert only after all padding is done
    #x_indices = [convert_to_indices(seq) for seq in x]

    x_one_hot = embed_one_hot(x, n_sym)
    return x, \
           np.expand_dims(y, axis=1), \
           x_one_hot