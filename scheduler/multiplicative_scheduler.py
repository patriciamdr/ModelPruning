from scheduler.abstract_scheduler import AbstractScheduler


class MultiplicativeScheduler(AbstractScheduler):
    """
    Multiply alpha by mu at every time step
    """
    def __init__(self, start=1, min=0, mu=0.9):
        super(MultiplicativeScheduler, self).__init__(start, min)
        self.mu = float(mu)
        # 1 instead of 0 to prevent start of objective selection with alpha=1
        self.last_ts = 1

    def step(self):
        val = self.start * self.mu ** self.last_ts
        self.last_ts += 1
        return val
