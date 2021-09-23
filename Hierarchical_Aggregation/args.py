import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('file_name', type=str, help='input file name')
    parser.add_argument('mode', choices=['local_topk', 'sketch', 'iceberg', 'mms', 'mms+oe'], default='mms+oe', help='encode mode')
    parser.add_argument('task', choices=['est', 'topk'], default='est', help='evaluate task')

    parser.add_argument('--min_ratio', type=float, default=1.5, help='minimum compression ratio')
    parser.add_argument('--max_ratio', type=float, default=5, help='maximum compression ratio')
    parser.add_argument('--ratio_step', type=float, default=0.5, help='the compression ratio changed per step')

    parser.add_argument('--max_worker', type=int, default=10000, help='maximum worker number')
    parser.add_argument('-c', '--cluster_num', type=int, default=100, help='the number of clusters')

    parser.add_argument('-g', '--global_k', type=int, default=10000, help='k in global top-k ')

    parser.add_argument('-t', '--test_time', type=int, default=10, help='test time')

    parser.add_argument('--row_num', type=int, default=3, help='row number in sketch')

    parser.add_argument('--outlier_thres', type=float, default=100, help='the threshold for eliminating outliers')

    return parser.parse_args()

