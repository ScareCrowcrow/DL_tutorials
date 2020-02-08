import sys

for line in sys.stdin:
    ss = line.strip().split("\t")
    user, item, score = ss
    print("%s\t%s\t%s" % (user, item, score))