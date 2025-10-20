class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print("best_loss updated from ", self.best_loss, " to ", val_loss)
            self.best_loss = val_loss
            self.counter = 0
        else:
            if self.verbose:
                print("Early stopper counter incremented from ", self.counter, " to ", self.counter + 1, "/", self.patience)
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
