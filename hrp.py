import argparse, numpy as np
from hrputils import calchrp

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-i', help='input csv file with returns', required=True)
my_parser.add_argument('-o', help='output csv file with hrp weights, defaults to weights.csv in the same folder', default='weights.csv')
args = my_parser.parse_args()

x = np.loadtxt(args.i,delimiter=',', dtype=float)
w = calchrp(x)
print(w)
np.array(w).tofile(args.o, sep=',')