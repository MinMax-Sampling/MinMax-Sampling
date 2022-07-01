inf = open('log_niid_local_50_mixed.txt', 'r')
lne = [l for l in inf]
cut = [eval(lne[24 * i + 23]) for i in range(10)]
for i in range(1, 11):
    print(cut[i % 10]['test_acc'])
