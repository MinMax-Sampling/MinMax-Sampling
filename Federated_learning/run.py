for k in range(2, 20 + 1, 2):
    for suffix in ['sketch', 'sampling', 'topk', 'nips']:
        f = open('log_dml_' + suffix + '_50_' + str(k) + '.txt')
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
        print(g[23], end = ' ')
    print()
