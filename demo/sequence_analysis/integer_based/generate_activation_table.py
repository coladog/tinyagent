import math
import struct
import numpy as np

from util import *


def q_mul(x, y):
    return int(x * y * FLOAT_RANGE) >> SCALE_BIT

def q_add(x, y):
    return x + y

def generate_sigmoid():
    def r_sigmoid(x):
        return -math.log((1 / x) - 1, math.e)

    sub = (1 / TABLE_RANGE)
    
    table = []
    bound = []
    for i in range(1, TABLE_RANGE):
        v = sub * i
        table.append(int(v / SCALE))
        bound.append(int(r_sigmoid(v) / SCALE))
    
    with open(SIGMOID_TABLE_OUTPUT, "wb") as f:
        for data in table:
            f.write(struct.pack("i", data))

    with open(SIGMOID_TABLE_INPUT, "wb") as f:
        for data in bound:
            f.write(struct.pack("i", data))

    print("[+] Generated table for integer-based Sigmoid activation:")
    print("[+] \t ", SIGMOID_TABLE_INPUT)
    print("[+] \t ", SIGMOID_TABLE_OUTPUT)

def generate_tanh():
    def atanh(x):
        z = (1 + x) / (1 - x)
        return math.log(z, np.e) / 2


    sub = (2 / TABLE_RANGE)
    
    table = []
    bound = []
    for i in range(1, TABLE_RANGE):
        v = -1 + sub * i
        table.append(int(v / SCALE))
        bound.append(int(atanh(v) / SCALE))
    
    with open(TANH_TABLE_OUTPUT, "wb") as f:
        for data in table:
            f.write(struct.pack("i", data))

    with open(TANH_TABLE_INPUT, "wb") as f:
        for data in bound:
            f.write(struct.pack("i", data))

    print("[+] Generated table for integer-based Tanh activation:")
    print("[+] \t ", TANH_TABLE_INPUT)
    print("[+] \t ", TANH_TABLE_OUTPUT)


if __name__ == "__main__":
    generate_sigmoid()
    generate_tanh()
