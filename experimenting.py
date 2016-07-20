import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from csxnet.high_utils import plot
from csxnet.nputils import import_from_csv
from csxnet.utilities import roots

data, labels, headers = import_from_csv(roots["tmp"] + "Factor.txt", encoding="utf8")


def plot3d(array):
    assert array.ndim == 2, "Can only plot 2 dimensional (matrix) data!"
    assert array.shape[1] > 2, "Need at least 3 columns!"
    if array.shape[1] > 3:
        print("Warning! Can only plot the first 3 columns, ignoring the rest!")
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(array[..., 0], array[..., 1], array[..., 2])
    plt.show()


def plot_sample(*samplenames):
    arrays = [data[argsample][0] for argsample in
              [np.where(labels == samplename)[0] for samplename in samplenames]]
    plot(*arrays)

if __name__ == '__main__':
    plot_sample("294A_", "298D_")
    plot3d(data[..., :3])
