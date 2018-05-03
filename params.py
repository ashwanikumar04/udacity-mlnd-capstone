class HyperParameter:
    def __init__(self, num_batches, batch_size, epoch, learning_rate, hold_prob, epoch_to_report=100):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hold_prob = hold_prob,
        self.epoch_to_report = epoch_to_report

    def __str__(self):
        return "epoch: " + str(self.epoch) + ", num_batches: " + str(self.num_batches) + ", batch_size: " + str(self.batch_size) + ", learning_rate: " + str(self.learning_rate) + ", hold_prob: " + str(self.hold_prob)
