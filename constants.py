from os import environ

THRESHOLD    = environ.get('THRESHOLD', -1)
LEARNINGRATE = environ.get('LEARNINGRATE', 0.01)
WEIGHTS = [0.5, 0.5, 1.0]