from csxnet.utilities import roots
from csxnet.nputils import import_from_csv


nir = roots["nir"] + "ntab.txt"
nir = import_from_csv(nir, labels=1, headers=True, sep="\t", end="\n")[0].reshape(69, 1800, 1)


def _1dautoencode(fanin):
    from keras.models import Sequential
    from keras.layers import Convolution1D, MaxPooling1D, UpSampling1D

    model = Sequential()
    model.add(Convolution1D(5, 31, input_shape=(fanin, 1), activation="relu", border_mode="same"))
    model.add(MaxPooling1D())
    model.add(UpSampling1D())
    model.add(Convolution1D(1, 31, activation="relu", border_mode="same"))
    model.compile("rmsprop", "mse")

    return model

ae = _1dautoencode(nir.shape[1])
ae.fit(nir, nir, batch_size=10)

print("asd")
