import numpy as np

idx = [10, 12, 14, 16, 18, 20, 2, 4, 6, 8]
r = 2


inf = open("log_niid_50_mixed.txt", "r")
lne = [l for l in inf]
cut = [eval(lne[i]) for i in range(40)]

for i in range(10):
    print(idx[i % 10])
for i in range(40):
    print(cut[i]['test_acc'])
