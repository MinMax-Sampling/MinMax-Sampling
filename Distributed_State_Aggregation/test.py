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


def run(vecs, args, run_args):
    if args.mode == 'sketch':
        est = CSVec(d=run_args['attri_num'],
                    c=run_args['col_num'], r=run_args['row_num'])
    elif args.mode == 'mms+oe':
        est = torch.zeros(run_args['attri_num']).cuda()
        w_r = torch.zeros(run_args['attri_num']).cuda()
        C = 0
    else:
        est = torch.zeros(run_args['attri_num']).cuda()
    real = torch.zeros(run_args['attri_num']).cuda()

    Prec = []
    MAE = []
    RMSE = []

    worker = 0
    for vec in vecs:
        if args.mode == 'mms+oe':
            _w_e, _w_r, _c = encode(args.mode, vec.cuda(), run_args)
            est += _w_e
            w_r += _w_r
            C += _c
        else:
            est += encode(args.mode, vec.cuda(), run_args)

        real += vec.cuda()

        worker += 1

        if worker % args.worker_step == 0:
            if args.mode == 'mms+oe':
                decode_est = outlier_eliminate(est, w_r, C, run_args)
            else:
                decode_est = decode(args.mode, est, run_args)

            if args.task == 'est':
                _mae, _rmse = evaluate(args.task, decode_est, real, run_args)
                MAE.append(_mae)
                RMSE.append(_rmse)
            else:
                _prec, _mae = evaluate(args.task, decode_est, real, run_args)
                Prec.append(_prec)
                MAE.append(_mae)

        if worker == args.max_worker:
            break

    return Prec, MAE, RMSE


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_args = dict()

    vecs, nonzero_num, run_args['attri_num'] = read_file(args.file_name)

    for ratio in numpy.arange(args.min_ratio, args.max_ratio+0.01, args.ratio_step):
        Prec = [0 for _ in range(len(vecs) // args.worker_step)]
        MAE = [0 for _ in range(len(vecs) // args.worker_step)]
        RMSE = [0 for _ in range(len(vecs) // args.worker_step)]

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
            Prec = [Prec[i] + _prec[i] / args.test_time for i in range(len(_prec))]
            MAE = [MAE[i] + _mae[i] / args.test_time for i in range(len(_mae))]
            RMSE = [RMSE[i] + _rmse[i] / args.test_time for i in range(len(_rmse))]

        for i in range(len(MAE)):
            if args.task == 'est':
                print(ratio, (i+1)*args.worker_step, MAE[i], RMSE[i])
            else:
                print(ratio, (i+1)*args.worker_step, Prec[i], MAE[i])
