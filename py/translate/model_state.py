
class model_state():
    def __init__(self):
        self.epochs = 0
        self.log = []

    def update(self, train_loss, val_bleu):
        self.epochs += 1
        log = {'epoch': self.epochs, 'train_loss': train_loss, 'val_bleu': val_bleu}
        self.log.append(log)
        return

    def next_epoch(self):
        return self.epochs + 1

    def get_log(self):
        return self.log
