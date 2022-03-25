from scheduler.abstract_scheduler import AbstractScheduler


class StepScheduler(AbstractScheduler):
    """
    Decays alpha continuously by step_size every time step
    """
    def __init__(self, start=1, min=0, step_size=0.1):
        super(StepScheduler, self).__init__(start, min)
        self.step_size = float(step_size)
        # 1 instead of 0 to prevent start of objective selection with alpha=1
        self.last_ts = 1

    def step(self):
        val = self.start - (self.step_size * self.last_ts)
        self.last_ts += 1
        return val
