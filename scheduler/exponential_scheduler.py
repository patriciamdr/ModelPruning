import numpy as np

from scheduler.abstract_scheduler import AbstractScheduler


class ExponentialScheduler(AbstractScheduler):
    """
    Decays alpha exponentially by decay_rate every time step
    """
    def __init__(self, start=1, min=0, decay_rate=0.1):
        super(ExponentialScheduler, self).__init__(start, min)
        self.decay_rate = float(decay_rate)
        # 1 instead of 0 to prevent start of objective selection with alpha=1
        self.last_ts = 1

    def step(self):
        val = self.start * np.e ** (-self.decay_rate * self.last_ts)
        self.last_ts += 1
        return val
