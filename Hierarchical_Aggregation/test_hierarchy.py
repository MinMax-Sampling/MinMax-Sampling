import torch
import numpy
from csvec import CSVec

from args import parse_args
from utils import encode, decode, outlier_eliminate
from evaluate import evaluate


def read_file(file_name):
    file = open(file_name)
    lines = file.readlines()
    file.close()

    vecs = []
    nonzero_num = 0
    for line in lines:
        vec = torch.Tensor(list(map(float, line.split())))
        vecs.append(vec)
        nonzero_num += vec.nonzero(as_tuple=False).size(0)

    return vecs, nonzero_num / len(vecs), vecs[0].size(0)


def aggregate(vecs, args, run_args, doOE=True):
    if args.mode == 'sketch':
        est = CSVec(d=run_args['attri_num'],
                    c=run_args['col_num'], r=run_args['row_num'])
        real = torch.zeros(run_args['attri_num']).cuda()
    elif args.mode == 'mms+oe':
        est = torch.zeros(run_args['attri_num']).cuda()
        w_r = torch.zeros(run_args['attri_num']).cuda()
        if 'C' in run_args:
            C = run_args['C']
        else:
            C = 0
        real = torch.zeros(run_args['attri_num']).cuda()
    else:
        est = torch.zeros(run_args['attri_num']).cuda()
        real = torch.zeros(run_args['attri_num']).cuda()

    for vec in vecs:
        if args.mode == 'mms+oe':
            _w_e, _w_r, _c = encode(args.mode, vec.cuda(), run_args)
            est += _w_e
            w_r += _w_r
            C += _c
        else:
            est += encode(args.mode, vec.cuda(), run_args)
        real += vec.cuda()

    if args.mode == 'mms+oe':
        if doOE:
            return outlier_eliminate(est, w_r, C, run_args), C, real
        else:
            return est, C, real
    else:
        return est, real


def run(vecs, args, run_args):
    if args.mode == 'sketch':
        est = CSVec(d=run_args['attri_num'],
                    c=run_args['col_num'], r=run_args['row_num'])
    elif args.mode == 'mms+oe':
        est = [torch.zeros(run_args['attri_num']).cuda() for _ in range(args.cluster_num)]
        C = 0
    else:
        est = [torch.zeros(run_args['attri_num']).cuda() for _ in range(args.cluster_num)]
    real = torch.zeros(run_args['attri_num']).cuda()

    Prec = MAE = RMSE = 0

    worker_per_cluster = args.max_worker // args.cluster_num
    for cluster_id in range(args.cluster_num):
        cluster_vecs = vecs[cluster_id*worker_per_cluster: (cluster_id+1)*worker_per_cluster]

        if args.mode == 'sketch':
            _est, _real = aggregate(cluster_vecs, args, run_args)
            est += _est
            real += _real
        elif args.mode == 'mms+oe':
            _est, _c, _real = aggregate(cluster_vecs, args, run_args, doOE=False)
            est[cluster_id] += _est
            C += _c
            real += _real
        else:
            _est, _real = aggregate(cluster_vecs, args, run_args)
            est[cluster_id] += _est
            real += _real

    if args.mode == 'sketch':
        aggregate_est = decode(args.mode, est, run_args)
    elif args.mode == 'mms+oe':
        run_args['C'] = C
        aggregate_est, _, _ = aggregate(est, args, run_args, doOE=True)
    else:
        aggregate_est, _ = aggregate(est, args, run_args)

    if args.task == 'est':
        MAE, RMSE = evaluate(args.task, aggregate_est, real, run_args)
    else:
        Prec, MAE = evaluate(args.task, aggregate_est, real, run_args)

    return Prec, MAE, RMSE


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_args = dict()

    vecs, nonzero_num, run_args['attri_num'] = read_file(args.file_name)

    for ratio in numpy.arange(args.min_ratio, args.max_ratio+0.01, args.ratio_step):
        Prec = MAE = RMSE = 0

        run_args['local_k'] = int(nonzero_num / ratio)
        if args.mode == 'sketch':
            run_args['col_num'] = run_args['local_k'] * 2 // args.row_num
            run_args['row_num'] = args.row_num
            run_args['global_k'] = run_args['attri_num']
        if args.mode == 'mms+oe':
            run_args['outlier_thres'] = args.outlier_thres
        if args.task == 'topk':
            run_args['global_k'] = args.global_k

        for _ in range(args.test_time):
            _prec, _mae, _rmse = run(vecs, args, run_args)
            Prec += _prec / args.test_time
            MAE += _mae / args.test_time
            RMSE += _rmse / args.test_time

        if args.task == 'est':
            print(ratio, args.max_worker, MAE, RMSE)
        else:
            print(ratio, args.max_worker, Prec, MAE)
