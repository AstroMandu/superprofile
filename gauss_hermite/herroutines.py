import numpy as np

def bound_to_lowspn(bound):
    return [bound[0], bound[1]-bound[0]]
def bound_to_loglowlogspn(bound):
    loglow = np.log(bound[0])
    logspn = np.log(bound[1])-loglow
    return [loglow, logspn]
def linmap(p, bound):
    low, spn = bound
    return (p - low) / spn
def logmap(p, logbound):
    loglow, logspn = logbound
    return (np.log(p) - loglow) / logspn
def invlinmap(u, bound):
    low, spn = bound
    return low + spn * u
def invlogmap(u, logbound):
    loglow, logspn = logbound
    return np.exp(loglow + logspn * u)