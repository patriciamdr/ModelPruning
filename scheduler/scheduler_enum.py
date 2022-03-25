from enum import Enum

class Scheduler(Enum):
    """
    Enum class for all types of schedulers.
    """
    Step = 'Step'
    Mult = 'Mult'
    Exp = 'Exp'
