import torch
from csvec import CSVec


def topk(vec, args):
    sort, idx = vec.sort(descending=True)
    _, topk_idx = sort[: args['local_k']], idx[: args['local_k']]

    return vec * topk_idx.bincount(minlength=vec.size(0)).cuda()


def sketch(vec, args):
    sketch = CSVec(d=vec.size(0), c=args['col_num'], r=args['row_num'])
    sketch.accumulateVec(vec)

    return sketch


def iceberg(vec, args):
    d = vec.size(0) / args['local_k']
    randv = torch.rand_like(vec).cuda()
    weight = vec / (vec + d)

    return torch.where(randv < weight, vec / weight, torch.zeros_like(weight))


def mms(vec, args):
    randv = torch.rand_like(vec).cuda()
    weight = (1 / randv - 1) * vec ** 2
    weight = torch.where(weight.isnan(), torch.zeros_like(weight), weight)

    sort, idx = weight.sort(descending=True)
    _, topk_idx = sort[: args['local_k']], idx[: args['local_k']]

    w_r = vec * topk_idx.bincount(minlength=vec.size(0)).cuda()

    C = sort[args['local_k']]

    w_e = w_r + C / w_r
    w_e = torch.where(w_e.isinf(), torch.zeros_like(vec), w_e)
    w_e = torch.where(w_e.isnan(), torch.zeros_like(vec), w_e)

    return w_e


def mms_for_oe(vec, args):
    randv = torch.rand_like(vec).cuda()
    weight = (1 / randv - 1) * vec ** 2
    weight = torch.where(weight.isnan(), torch.zeros_like(weight), weight)

    sort, idx = weight.sort(descending=True)
    _, topk_idx = sort[: args['local_k']], idx[: args['local_k']]

    w_r = vec * topk_idx.bincount(minlength=vec.size(0)).cuda()

    C = sort[args['local_k']]

    w_e = w_r + C / w_r
    w_e = torch.where(w_e.isinf(), torch.zeros_like(vec), w_e)
    w_e = torch.where(w_e.isnan(), torch.zeros_like(vec), w_e)

    return w_e, w_r, C


def encode(func_name, vec, args):
    return {
        'local_topk': topk,
        'sketch': sketch,
        'iceberg': iceberg,
        'mms': mms,
        'mms+oe': mms_for_oe,
    }[func_name](vec, args)



def no_decode(vec, args):
    return vec


def unsketch(cs, args):
    return cs.unSketch(k=args['global_k'])


def outlier_eliminate(w_e, w_r, C, args):
    x = w_e ** 3 / (w_e ** 2 + C) / w_r
    x = torch.where(x.isinf(), torch.zeros_like(x), x)
    x = torch.where(x.isnan(), torch.zeros_like(x), x)

    return torch.where(x > args['outlier_thres'], torch.zeros_like(w_e), w_e)


def decode(func_name, vec, args):
    return {
        'local_topk': no_decode,
        'sketch': unsketch,
        'iceberg': no_decode,
        'mms': no_decode,
    }[func_name](vec, args)
