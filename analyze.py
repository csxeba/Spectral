import os

import numpy as np

from csxnet.utilities import roots
from csxnet.nputils import subsample, avg2pool
from csxnet.high_utils import autoencode


datapath = roots["nir"]


def extract_data():

    def from_file_to_array(path):
        f = open(path)
        data = f.read().split("\n")
        data = [d.split(",") for d in data if d]
        return np.array(data).T

    def extract_header(flz):
        heads = [from_file_to_array(fl)[0] for fl in flz]
        truth = True
        for head in heads[1:]:
            truth = truth or np.all(np.equal(head, heads[0]))
        assert truth, "Headers are different..."
        return heads[0]

    os.chdir(datapath)
    files = [fl for fl in os.listdir(".") if fl[-4:] == ".dpt"]
    header = extract_header(files)
    dataz = [array[1] for array in map(from_file_to_array, files)]
    dataz = np.vstack(dataz).astype("float64")
    labels = [fl[:-43] for fl in files]
    if "fruits.txt" not in os.listdir("."):
        export_to_file("fruits.txt", dataz, labels, header, pkl=True)
    return dataz, labels, header


def export_to_file(path, data, labels, headers, pkl=False):
    outchain = ""
    if headers is not None:
        outchain = "MA\t"
        outchain += "\t".join(headers) + "\n"
    for i, samplename in enumerate(labels):
        outchain += str(samplename) + "\t"
        outchain += "\t".join(data[i].astype("<U11")) + "\n"
    with open(path, "w", encoding="utf8") as outfl:
        outfl.write(outchain.replace(".", ","))
        outfl.close()
    if pkl:
        data.dump("data.npa")
        np.array(labels).dump("labels.npa")
        if headers is not None:
            headers.dump("headers.npa")


def do_pca(matrix):
    from sklearn.decomposition import PCA
    pca = PCA(whiten=True)
    pca.fit(matrix)
    transformed = pca.transform(matrix)
    return transformed


if __name__ == '__main__':
    data, labels, headers = extract_data()
    autoencoded = autoencode(data, 30)
    export_to_file("E:/tmp/autoencoded01.csv", autoencoded, labels, None)
