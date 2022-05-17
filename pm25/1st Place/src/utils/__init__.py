import os
import time
import numpy as np

def check_existance(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return True

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Timer:
    def __init__(self):
        self.cur_state = time.time()
        
    def _format_time(self, seconds):
        """
        Formats time in human readable form

        Args:
            seconds: seconds passed in a process
        Return:
            formatted string in form of MM:SS or HH:MM:SS
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        result = ''
        _h = ('0' + str(h)) if h < 10 else str(h)
        result += (_h + ' hr ') if h > 0 else ''
        _m = ('0' + str(m)) if m < 10 else str(m)
        result += (_m + ' min ') if m > 0 else ''
        _s = ('0' + str(s)) if s < 10 else str(s)
        result += (_s + ' sec')
        return result
    
    def beep(self, prefix="Time Elapsed: "):
        elapsed_time = time.time() - self.cur_state
        self.cur_state = time.time()
        return prefix + self._format_time(elapsed_time)
    
    def reset(self):
        self.cur_state = time.time()
        

# Metrics
def mse(preds, targets):
    return np.mean((preds - targets) ** 2)

def rmse(preds, targets):
    return np.sqrt(mse(preds, targets))

def rsquared(preds, targets):
    target_mean = targets.mean()
    ss_tot = np.sum((targets - target_mean) ** 2)
    ss_res = np.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def compute_metrics(preds, targets):
    return {
        'mse': mse(preds, targets),
        'rmse': rmse(preds, targets),
        'r2': rsquared(preds, targets)
    }