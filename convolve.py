from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense

from csxnet.utilities import roots
from csxnet.nputils import import_from_csv


hsc1 = roots["nir"] + "ntab.txt"
hsc1, _, __ = import_from_csv(hsc1, labels=1, headers=True, sep="\t", end="\n")
