import numpy as np

from .util import time_to_frames, get

OUTLIER_REMOVAL_CONSTANT = np.log(0.005)

def compose(x, config=dict()):
    '''
    Applies specified normalization techniques onto given values.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Sequence of values which are normalized.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. For parameters
        used in normalization functions themselves, check their description. Following parameters are used in this
        function:

        * 'normalization.techniques'

    Returns
    -------
    x' : np.ndarray [shape=(n,)]
        Sequence of normalized values.
    '''

    techniques = get(config, 'normalization.techniques')

    for technique in techniques:
        if technique == 'outlier_removal':
            x = outlier_removal(x, config)
        elif technique == 'moving_average':
            x = moving_average(x, config)
        elif technique == 'min_max':
            x = min_max(x, config)
        else:
            raise ValueError('Only "outlier_removal", "moving_average", or "min_max" are supported')

    return x

def outlier_removal(x, config=dict()):
    '''
    Shrinks values of outliers to a computed threshold values, effectively removing the outliers.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Set of values in which outliers are removed.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. This function use
        no parameters.

    Returns
    -------
    x' : np.ndarray [shape=(n,)]
        Set of values with outliers removed.
    '''

    threshold = -np.mean(x) * OUTLIER_REMOVAL_CONSTANT
    return np.minimum(x, threshold)

# moving average implemented as convolution
def moving_average(x, config=dict()):
    '''
    Applies moving average to given sequence of values.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Sequence of values on which moving average is applied.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'normalization.moving_average.window_length'

    Returns
    -------
    x' : np.ndarray [shape=(n,)]
        Sequence of averaged values.
    '''

    window_length = time_to_frames(get(config, 'normalization.moving_average.window_length'), config)
    window = np.ones(window_length) / window_length
    return np.convolve(x, window, 'same')

def min_max(x, config=dict()):
    '''
    Scales values to [0, 1] range linearly.

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        Set of values which are scaled.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. This function use
        no parameters.

    Returns
    -------
    x' : np.ndarray [shape=(n,)]
        Set of scaled values.
    '''

    minimum = np.min(x)
    maximum = np.max(x)
    return (x - minimum) / (maximum - minimum)
