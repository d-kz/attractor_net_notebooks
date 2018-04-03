import pickle
import datetime
import scipy as sc
import numpy as np
from sklearn.metrics.cluster import mutual_info_score



class MutInfSaver():
    def __init__(self):
        self.losses_task = []
        self.h_inits = [] # h_init, all attractor iterations, h_next
        self.h_attractors = []
        self.h_finals = []
        self.alosss = []
        self.train_accs = []
        self.test_accs= []
        
    def update(self, loss_task, aloss, train_acc, test_acc, h_init, h_attractor, h_final):
        self.losses_task.append(loss_task)
        self.h_inits.append(h_init)
        self.h_attractors.append(h_attractor)
        self.h_finals.append(h_final)
        self.alosss.append(aloss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)

    def split_ids(self):
        "splits the array into separate sessions and normalizes each session by its maximum id (epoch number)"
        a = self.losses_task
        ids = []
        for i in range(len(a)):
            if a == 0.0:
                ids.append(i)
        return ids    
    
class WeightSaver():
    def __init__(self):
        self.epoch_number_history = []
        self.losses_att = []
        self.losses_task = []
        self.weights_history = []
        self.acc_history = []
        self.h_history = []
        self.bias_history = []
        self.scaling_const_history = []
        self.bias_lambda_history = []
        self.entropies = []
        
        self.directory = "experiments/weight_pickles/"
    
    def update(self, epoch_number, loss_att, loss_task, W_att, b_att, scaling_const, acc, h_seq):
        self.epoch_number_history.append(epoch_number)
        
        self.losses_att.append(loss_att)
        self.losses_task.append(loss_task)
        
        self.weights_history.append(W_att)
        self.bias_history.append(b_att)
        self.scaling_const_history.append(scaling_const)
        
        self.acc_history.append(acc)
        self.h_history.append(h_seq)
        
        
    def update_conservative(self, epoch_number, loss_att, loss_task, acc, entropy):
        self.epoch_number_history.append(epoch_number)
        self.losses_att.append(loss_att)
        self.losses_task.append(loss_task)
        self.acc_history.append(acc)
        self.entropies.append(entropy)

    def get_hashmap_format(self):
        hashmap = {
            "losses_att"        :   self.losses_att,
            "losses_task"       :   self.losses_task, 
            "weights_history"   :   self.weights_history, 
            "acc_history"       :   self.acc_history, 
            "h_history"         :   self.h_history, 
            "bias_history"      :   self.bias_history,
            "bias_lambda"       :   self.bias_lambda_history
        }
        
        return hashmap
    
    def pickle_history(self, ops, comment):
        element_map = self.get_hashmap_format()
        title = self.directory + "({})[Model<{}>(attractor<{}>, noise<{}>), Problem<{}>]__{}.pickle".format(datetime.date.today(),
                                                           ops['model_type'], 
                                                           str(ops['attractor_dynamics']),
                                                           ops['attractor_noise_level'],
                                                           ops['problem_type'],
                                                           comment)
        with open(title, 'wb') as handle:
            pickle.dump(element_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved successfully")
        
    def unpickle_history(self, title, rewrite=False):
        title = self.directory + title
        with open(title, 'rb') as handle:
            element_map = pickle.load(handle)
        
        if rewrite:
            self.losses_att = element_map["losses_att"]
            self.losses_task = element_map["losses_task"]
            self.weights_history = element_map["weights_history"]
            self.acc_history = element_map["acc_history"]
            self.h_history = element_map["h_history"]
            self.bias_history = element_map["bias_history"]
            
        return element_map


def compute_entropy_fullvec(h_final, ops, n_bins=10):
    if h_final == []:
        return 0.0  # for matching a space between new runs

    h_flat = h_final.reshape(-1, ops['hid'])
    # put values into bins
    bins = np.array([-1.0 + 2.0 * i / n_bins for i in range(n_bins + 1)])
    h_digitized = np.digitize(h_flat, bins)
    # compute probability for each bin
    neuron_distn = h_digitized
    # axis=0 will compute unique vectors instead of by element
    _, pdf = np.unique(neuron_distn, return_counts=True, axis=0)  # returns sorted by unique value
    neurons_entropy = sc.stats.entropy(pdf)
    return neurons_entropy


def get_mut_inf_for_vecs(A, B, n_bins=10):
    """
    return the average mut_inf between all neuron pairs of 2 vectors given
    """
    m = A.shape[1]
    n = B.shape[1]
    bins = np.array([-1.0 + 2.0 * i / n_bins for i in range(n_bins + 1)])
    A_digitized = np.digitize(A, bins)
    B_digitized = np.digitize(B, bins)

    total_mut_inf = 0.0
    for i in range(m):
        for j in range(n):
            #             print(A[i], A_digitized[i])
            #             print(mutual_info_score(A_digitized[i], B_digitized[j]))
            total_mut_inf += mutual_info_score(A_digitized[:, i], B_digitized[:, j])
    return total_mut_inf / (m * n)


def flat_mutual_inf(h_init, h_atts, h_final, ops):
    """
    returns an array of of mutual information values between initial hidden vector
    and all consecutive attractor state vectors
    - each "distribution" of a neuron is an observation for one sequence
    """
    if h_init == []:
        return []  # for matching a space between new runs
    # flatten over the sequence & batches
    H_target = h_init.reshape(-1, ops['hid'])
    Hs = [H_target]
    for h_att in h_atts:
        Hs.append(h_att.reshape(-1, ops['h_hid']))
    Hs.append(h_final.reshape(-1, ops['hid']))

    mut_infs = []
    for H in Hs:
        mut_inf = get_mut_inf_for_vecs(H_target, H)
        mut_infs.append(mut_inf)
    return mut_infs


def compute_avg_entropy_vec(h_final, ops, n_bins=10):
    if h_final == []:
        return 0.0  # for matching a space between new runs

    h_flat = h_final.reshape(-1, ops['hid'])

    # put values into bins
    bins = np.array([-1.0 + 2.0 * i / n_bins for i in range(n_bins + 1)])
    h_digitized = np.digitize(h_flat, bins)
    # compute probability for each bin
    neurons_entropy = []
    for neuron_i in range(h_digitized.shape[1]):
        neuron_distn = h_digitized[:, neuron_i]
        #         pdf = np.bincount(neuron_distn)/len(neuron_distn)
        _, pdf = np.unique(neuron_distn, return_counts=True)  # returns sorted by unique value
        neurons_entropy.append(sc.stats.entropy(pdf))
    return np.average(neurons_entropy)

