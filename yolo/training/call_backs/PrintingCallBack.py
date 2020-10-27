import tensorflow.keras as ks


class Printer(ks.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._step = 1
        self._dump = 0
        self._avg_dict = {}
        self._print_frq = 1
        self._end = "\n"
        return

    def _print_me(self, batch, logs=None):
        if logs != None:
            for key in logs.keys():
                if key not in self._avg_dict.keys():
                    self._avg_dict[key] = logs[key]
                else:
                    self._avg_dict[key] += logs[key]

            if self._dump == 0:
                printer = f"step: {self._step} \t||"
                printer += " %s: %0.3f \t" % ("loss", self._avg_dict["loss"] /
                                              self._step)
                for key in sorted(self._avg_dict.keys()):
                    if key != "loss":
                        printer += " %s: %0.3f \t" % (
                            key, self._avg_dict[key] / self._step)
                print(printer, end=self._end, flush=True)
                #self._avg_dict = dict()
        self._step += 1
        self._dump = (self._dump + 1) % self._print_frq
        return

    def on_train_batch_end(self, batch, logs=None):
        self._print_me(batch, logs=logs)
        return

    def on_test_batch_end(self, batch, logs=None):
        self._print_me(batch, logs=logs)
        return

    def on_train_epoch_begin(self, epoch, logs):
        self._step = 1
        self._avg_dict = dict()
        print(epoch)

    def on_test_epoch_begin(self, epoch, logs):
        self._step = 1
        self._avg_dict = dict()
        print(epoch)
