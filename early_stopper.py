################ EarlyStopper class #####################################################

class EarlyStopper():
    def __init__(self, patience_max, disp_epoch, min_delta = 0.00):
        self.best_val_err = 1e10
        self.patience = 0  # our patience
        self.patience_max = patience_max
        self.display_epoch = disp_epoch
        self.min_delta = min_delta
        self.best_train_acc = 0.
        self.best_test_acc = 0.

    def update(self, current_val_err, current_train_acc, current_test_acc):
        if self.best_val_err > current_val_err:
            self.best_val_err = current_val_err
            self.best_test_acc = current_test_acc
            self.best_train_acc = current_train_acc
            self.patience = 0
        elif abs(self.best_val_err - current_val_err) > self.min_delta:
            self.patience += 1

    def patience_ran_out(self):
        if self.patience*self.display_epoch > self.patience_max:
            return True
        else:
            False