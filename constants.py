from os import environ

### EDIT ME HERE ###
#                  
w1 = 0.5
w2 = 0.5
w0 = 1.0
_threshold = -1
_learningrate = 0.01
_epoch = 100
#
####################

# set-up environ
THRESHOLD    = float(environ.get('THRESHOLD', _threshold))
LEARNINGRATE = float(environ.get('LEARNINGRATE', _learningrate))
WEIGHTS = [float(environ.get('W1', w1)), float(environ.get('W2', w2)), float(environ.get('W0', w0))]
EPOCH = int(environ.get('EPOCH', _epoch))