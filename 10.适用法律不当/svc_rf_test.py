import json
import os
from utils import util
import multiprocessing

from predictor import Predictor

data_path = "./data/test.tsv"  # The directory of the input data
output_dir = "./output/%s/" % util.MODEL  # The directory of the output data
output_path = output_dir + "test.txt"


if __name__ == "__main__":
    user = Predictor()
    cnt = 0


    def get_batch():
        v = user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v


    def solve(fact):
        result = user.predict_my(fact)
        return result


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    inf = open(data_path, "r")
    ouf = open(output_path, "w")

    fact = []
    label = []
    inf.readline()
    line = inf.readline()

    while line:
        fact.append(line.strip("\n").split('\t')[1])
        label.append(line.strip("\n").split('\t')[2])
        if len(fact) == get_batch():
            result = solve(fact)
            cnt += len(result)
            for i,x in enumerate(result):
                print(str(x[0])+","+label[i], file=ouf)
            fact = []
            label = []
        line = inf.readline()

    if len(fact) != 0:
        result = solve(fact)
        cnt += len(result)
        for i, x in enumerate(result):
            print(str(x[0]) + "," + label[i], file=ouf)
        fact = []
        label = []

    inf.close()
    ouf.close()
if util.DEBUG:
    print("DEBUG: prediction work finished.")

