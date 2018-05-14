from . import lib

def estimate(filepath, config=dict()):
    '''
    Runs the whole pipeline to estimate song boundaries from given audio file.

    Parameters
    ----------
    filepath : string
        Filepath of the audio file.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. No parameters are
        used in this function directly. For other parameters used in stage functions themselves, check their
        description.

    Returns
    -------
    songs : np.ndarray [shape=(n_songs,)]
        List of estimated songs with their start and end time annotations in seconds.
    '''

    audio = lib.audio.load(filepath, config)

    spectrum = lib.spectrum.compute(audio, config)

    rms = lib.feature.rms(spectrum, config)
    centroid = lib.feature.centroid(spectrum, config)
    flux = lib.feature.flux(spectrum, config)
    flatness = lib.feature.flatness(spectrum, config)

    rms = lib.normalization.compose(rms, config)
    centroid = lib.normalization.compose(centroid, config)
    flux = lib.normalization.compose(flux, config)
    flatness = lib.normalization.compose(flatness, config)

    llr = lib.segmentation.llr(rms, centroid, flux, flatness, config)
    boundaries = lib.segmentation.get_boundaries(llr, config)

    songs = lib.classification.classify(rms, centroid, flux, flatness, boundaries, config)

    songs = lib.postprocessing.process(songs, config)

    return songs

def to_time_annotations(songs, both=True):
    '''
    Converts songs' boundaries from seconds to time format representation.

    Parameters
    ----------
    songs : np.ndarray [shape=(n_songs,)]
        List of songs with their start and end time annotations in seconds.

    both : bool
        Determines if both boundaries, i.e. start and end, are printed. If `False`, only start times are printed.

    Returns
    -------
    songs : string
        Songs' boundaries formatted into string.
    '''

    output = ''

    if both:
        for start, end in songs:
            output += lib.util.seconds_to_string(start)
            output += ' '
            output += lib.util.seconds_to_string(end)
            output += '\n'
    else:
        for start, _ in songs:
            output += lib.util.seconds_to_string(start)
            output += '\n'

    return output
