import sys
import math


"""
对item做相似度，对item进行归一化

将 item-user-score 转换为 user-item-score
"""

cur_item = None
user_score_list = []

for line in sys.stdin:
    ss = line.strip().split("\t")
    item, user, score = ss

    if not cur_item:
        cur_item = item

    if item != cur_item:
        sum = 0.0
        for tuple in user_score_list:
            u, s = tuple
            sum += pow(s, 2)
        sum = math.sqrt(sum)

        for tuple in user_score_list:
            u, s = tuple
            print("%s\t%s\t%s" % (u, cur_item, float(s/sum)))
        user_score_list = []
        cur_item = item

    user_score_list.append((user, float(score)))

sum = 0.0
for tuple in user_score_list:
    u, s = tuple
    sum += pow(s, 2)
sum = math.sqrt(sum)

for tuple in user_score_list:
    u, s = tuple
    print("%s\t%s\t%s" % (u, cur_item, float(s/sum)))
