import pickle
import datetime

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
        
        
    def update_conservative(self, epoch_number, loss_att, loss_task, acc):
        self.epoch_number_history.append(epoch_number)
        self.losses_att.append(loss_att)
        self.losses_task.append(loss_task)
        self.acc_history.append(acc)

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
    