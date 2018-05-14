from segson import estimate, to_time_annotations

import sys
import os.path

def error(message):
    sys.stderr.write(message + '\n')
    sys.exit(1)


if __name__ == '__main__':
    filepath = None
    arguments = []
    config_params = []

    for system_arg in sys.argv[1:]:
        if os.path.isfile(system_arg):
            filepath = system_arg
        elif system_arg.startswith('-'):
            arguments.append(system_arg)
        else:
            config_params.append(system_arg)

    if filepath is None:
        error('Input file not found')

    config = dict()

    for param in config_params:
        split = param.split('=')

        if len(split) != 2:
            error('Parameter "{}" is not in valid format. Use param=value.'.format(param))

        try:
            if '.' in split[1]:
                config[split[0]] = float(split[1])
            else:
                config[split[0]] = int(split[1])
        except ValueError:
            config[split[0]] = split[1]

    songs = estimate(filepath, config)

    formatted = to_time_annotations(songs, '--only-start' not in arguments)

    sys.stdout.write(formatted)
