import math
import numpy as np

from KDEpy import FFTKDE

from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn import metrics
from scipy.signal import find_peaks, peak_widths

def get_vals_in_range(m, lower_bound=0, upper_bound=0.99):
    '''
    Given a numpy array `m`, return a list with all values in `m` that are within range specified by `lower_bound` and `upper_bound` 
    '''
    m_list = m.flatten().tolist()
    m_list = [val for val in m_list if  lower_bound < val < upper_bound]
    return m_list 

def get_kde(vals, bandwidth=1, bandwidth_multiplier=1, num_samples=1000, boundary_proportion=0.1, dynamic_num_samples=False, log_base=10, bandwidth_selection=None, kernel='gaussian', bandwidth_based_on_samples=False):
    '''
    Given a list of uniformly separated values `vals` estimate density distribution using KDE

    `bandwidth_selection`: must be one of [None, 'ISJ']. If specified then `bandwidth_multiplier` is ignored
    and KDEpy is used. 

    Returns 4 items:
        1. the density values list
        2. the samples at which the density is estimated at
        3. the samples lower bound
        4. the samples upper bound
    '''

    # Compute the range of `vals` and assign upper and lower boundaries for the samples 
    vals_range = max(vals) - min(vals)

    if vals_range == 0:
        # If all values in `vals` have the same value then set the lower bound to 0 and the upper bound to 1
        samples_lower_bnd = 0
        samples_upper_bnd = 1
    else:
        samples_lower_bnd = min(vals) - boundary_proportion*vals_range
        samples_upper_bnd = max(vals) + boundary_proportion*vals_range

    if dynamic_num_samples:
        # The number of samples are dynamical set depending on the `vals_range`
        num_samples = math.ceil(num_samples / (1 - math.log(samples_upper_bnd - samples_lower_bnd, log_base)))

    # Compute the KDE curve
    samples = np.linspace(samples_lower_bnd, samples_upper_bnd, num=num_samples)

    if bandwidth_selection=='ISJ':
        try:
            density = FFTKDE(kernel=kernel, bw='ISJ').fit(vals).evaluate(samples)
            return density, samples, samples_lower_bnd, samples_upper_bnd
        except:
            print("Failed to perform KDE using fixed bandwidth approach")

    points = np.array(np.array(vals)).reshape(-1,1)
    if bandwidth_based_on_samples:
        bandwidth=((samples_upper_bnd - samples_lower_bnd)/num_samples)*bandwidth_multiplier
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(points)
    density = np.exp(kde.score_samples(samples.reshape(-1,1)))

    return density, samples, samples_lower_bnd, samples_upper_bnd