"""
First used in 3.1.2 in <d2l>
"""

import time

import numpy as np

class Timer:
    """Record multiple running times."""
    def __init__(self, round=None):
        self.times = []
        self.round = round
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """
        Stop the timer and record the time in a list.
        Return the last time.
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        if self.round:
            return round(sum(self.times) / len(self.times), self.round)
        else:
            return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """
        Return the accumuated times.
        Each item is the sum of the corresponding item and all the items preceding it
        """
        return np.array(self.times).cumsum().tolist()
