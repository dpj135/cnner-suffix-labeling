import logging
import pickle
import time
from typing import Tuple

import numpy
from sympy import false


def get_logger(dataset):
    pathname = "./train_logs/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs:numpy.array, entities:numpy.array, length:numpy.array):
    ent_r, ent_p, ent_c = 0, 0, 0
    confidence=0.3
    decode_entities = []

    for instance, ent_set, l in zip(outputs, entities, length):
        results=[]
        #print('xxxxxxx\n',instance)
        for end in range(l):
            for start in range(end+1):
                type_id=numpy.argmax(instance[start][end])
                if type_id>0 and instance[start][end][type_id]>confidence:
                    results.append( ( (start, end) , type_id, instance[start][end][type_id]) )
        results.sort(key=lambda x: x[2],reverse=True)
        def is_overlapping(x,y):
            if x[0]>y[0]:
                x,y = y,x
            if x[0]<y[0] and x[1]<y[1] and x[1]>=y[0]:
                return True
            return False
        predicts=[]
        for i in range(len(results)):
            is_ok=True
            for j in range(i):
                if is_overlapping(results[i][0],results[j][0]):
                    is_ok=False
                    break
            if is_ok==True:
                predicts.append( (list(range(results[i][0][0],results[i][0][1]+1)) , results[i][1]) )

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        #print('\n---------\npredicts: ', predicts)
        #print('\n---------\nent_set: ', ent_set)
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
        #print('\n---------\nresults: ', results)


    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
