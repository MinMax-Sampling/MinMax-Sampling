gg = []
for k in range(20, 20 + 1, 20):
    for suffix in ['sketch', 'sampling', 'local', 'nips', 'uncompressed']:
        if (suffix == 'uncompressed'):
            f = open('log_niid_baseline_50' + '.txt')
        else:
            f = open('log_niid_' + suffix + '_50_' + str(k) + '.txt')
        N = 0
        g = []
        for line in f:
            x = line.find('test_loss')
            y = line.find(',', x)
            if x != -1:
                N += 1
                g.append(line[x+12:y])
        for i in range(24 - N):
            g.append(0)
        gg.append(g)
    print()

for i in range(24):
    for j in range(5):
        print(gg[j][i], end = ' ')
    print()
