import sys
import random
import argparse

parser = argparse.ArgumentParser(description='Generate 2^X by 2^Y matrix')
parser.add_argument('seed', type=int, help='Seed for the random number generator')
parser.add_argument('X', type=int, help='Power of two for number of rows')
parser.add_argument('Y', type=int, help='Power of two for number of columns')
args = parser.parse_args()


random.seed(args.seed)

for row in range(2 ** args.X):
    print ("\t".join(str(random.random()) for x in range(2 ** args.Y)))