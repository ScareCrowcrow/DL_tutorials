import sys


for line in sys.stdin:
    i_a, i_b, score = line.strip().split("\t")
    print("%s\t%s" % (i_a + "^A" + i_b, score))