import os

import numpy as np


def merge_datapoint_files(rootdir):

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

    os.chdir(rootdir)
    files = [fl for fl in os.listdir(".") if fl[-4:] == ".dpt"]
    header = extract_header(files)
    dataz = [array[1] for array in map(from_file_to_array, files)]
    dataz = np.vstack(dataz).astype("float64")
    labels = [fl[:-43] for fl in files]
    return dataz, labels, header
