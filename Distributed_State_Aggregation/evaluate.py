import torch
import numpy


def evaluate_all(est, real, args):
    MAE = (est - real).abs().sum() / real.size(0)

    RMSE = (((est - real) ** 2).sum() / real.size(0)) ** 0.5

    return MAE.item(), RMSE.item()


def evaluate_topk(est, real, args):
    _, est_idx = est.topk(args['global_k'])
    _, real_idx = real.topk(args['global_k'])

    inter_num = numpy.intersect1d(
        est_idx.cpu().numpy(), real_idx.cpu().numpy()).size
    Prec = inter_num / args['global_k']

    est_topk = est * est_idx.bincount(minlength=est.size(0)).cuda() * real_idx.bincount(minlength=est.size(0)).cuda()
    real_topk = real * real_idx.bincount(minlength=real.size(0)).cuda()

    MAE = (est_topk - real_topk).abs().sum() / args['global_k']

    return Prec, MAE.item()


def evaluate(func_name, est, real, args):
    return {
        'est': evaluate_all,
        'topk': evaluate_topk,
    }[func_name](est, real, args)
