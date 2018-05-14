from .util import get

def process(songs, config=dict()):
    '''
    Process identified songs.

    Parameters
    ----------
    songs : np.ndarray [shape=(n_songs,)]
        List of songs with their start and end time annotations in seconds.

    config : dict
        Configuration dictionary. For full list of parameters with their description, see README file. Following
        parameters are used in this function:

        * 'postprocessing.minimum_threshold'
        * 'postprocessing.closest_neighbor'

    Returns
    -------
    songs : np.ndarray [shape=(n_songs_postprocessed,)]
        List of postprocessed songs with their start and end time annotations in seconds.
    '''

    minimum_threshold = get(config, 'postprocessing.minimum_threshold')
    closest_neighbor = get(config, 'postprocessing.closest_neighbor')

    if len(songs) == 2:
        left = songs[1][0] - songs[0][1]

        if songs[1][1] - songs[1][0] < minimum_threshold and left < closest_neighbor:
            return [(songs[0][0], songs[1][1])]
    else:
        for index in range(1, len(songs) - 1):
            if songs[index][1] - songs[index][0] < minimum_threshold:
                right = songs[index + 1][0] - songs[index][0]

                if songs[index - 1] is not None:
                    left = songs[index][0] - songs[index - 1][1]
                else:
                    left = right + 1

                if left <= right and left < closest_neighbor:
                    songs[index - 1] = (songs[index - 1][0], songs[index][1])
                    songs[index] = (songs[index - 1][0], songs[index][1])
                elif right < left and right < closest_neighbor:
                    songs[index] = (songs[index][0], songs[index + 1][1])
                    songs[index + 1] = (songs[index][0], songs[index + 1][1])
                else:
                    songs[index] = None

    return sorted(set(filter(lambda song: song is not None, songs)), key=lambda song: song[0])
