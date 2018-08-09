class EarlyStopper():
    def __init__(self, patience_max, disp_epoch, min_delta = 0.00):
        self.best = 1e10
        self.patience = 0  # our patience
        self.patience_max = patience_max
        self.display_epoch = disp_epoch
        self.min_delta = min_delta

    def update(self, current):
        if self.best > current:
            self.best = current
            self.patience = 0
        elif abs(self.best - current) > self.min_delta:
            self.patience += 1

    def patience_ran_out(self):
        if self.patience*self.display_epoch > self.patience_max:
            return True
        else:
            False


# initialize gradient buckets for all variables of interest as tf.Variable()
accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in attr_net_parameters]
# operation to "flush" (zero out) all gradient buckets
zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
# operation for gradient computation
numerator_gradient_op = optimizer_attr.compute_gradients(numerator, var_list=attr_net_parameters)

# add (assign_add) computed gradients to each bucket
numerator_accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(numerator_gradient_op)]

# finally an actual training step
numerator_train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(numerator_accum_ops)])

sess.run(zero_ops) # flusho out our containers for gradients
for i in xrange(n_minibatches):
    # add gradients to buckets
	sess.run(numerator_accum_ops, feed_dict=dict(X: Xs[i], y: ys[i]))
# apply gradient to update the weights
sess.run(numerator_train_step)
