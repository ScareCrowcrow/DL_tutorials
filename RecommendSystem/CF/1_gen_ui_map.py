import sys

"""
将 user--item--score 转换为 item--user--score格式
"""
for line in sys.stdin:
    ss = line.strip().split("\t")
    if len(ss) != 3:
        continue
    user, item, score = ss
    print("%s\t%s\t%s" % (item, user, score))
