import librosa

from .util import get

def load(filepath, config=dict()):
    '''
    Loads an audio file (or just a part of it) located in given filepath. It is resampled and transformed into mono
    signal.

    Parameters
    ----------
    filepath : string
        Filepath of the audio file.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'audio.sampling_rate'
        * 'audio.offset'
        * 'audio.duration'

    Returns
    -------
    audio : np.ndarray [shape=(n_samples,)]
        Retrieved audio signal.
    '''

    sr = get(config, 'audio.sampling_rate')
    offset = get(config, 'audio.offset')
    duration = get(config, 'audio.duration')

    y, _ = librosa.load(filepath, sr=sr, offset=offset, duration=duration, res_type='kaiser_fast')
    return y
