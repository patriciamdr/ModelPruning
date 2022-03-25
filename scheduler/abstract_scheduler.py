from abc import abstractmethod


class AbstractScheduler:
    """
    Abstract superclass for all types of alpha-schedulers
    """

    def __init__(self, start, min):
        self.start = start
        self.min = min

    @abstractmethod
    def step(self, *args, **kwargs):
        """
        :return: Next alpha value
        """
        # To be implemented in subclasses
        raise NotImplementedError('subclasses must override step()!')
