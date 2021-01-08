from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from utils.image import get_affine_transform, affine_transform, transform_preds_with_trans


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    # perform nms on heatmaps
    heat = _nms(heat)  # 检测当前热点的值是否比周围的八个近邻点(八方位)都大(或者等于)

    scores, inds, clses, ys, xs = _topk(heat, K=K)  # 取100个这样的点
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def generic_decode(output, K=100):
    heat = output['hm']
    batch, cat, height, width = heat.size()

    heat = _nms(heat)
    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

    clses = clses.view(batch, K)
    scores = scores.view(batch, K)
    cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {'scores': scores, 'clses': clses.float(),
           'xs': xs0, 'ys': ys0, 'cts': cts}

    reg = output['reg']
    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]

    wh = output['wh']
    wh = _tranpose_and_gather_feat(wh, inds)  # B x K x (F)
    # wh = wh.view(batch, K, -1)
    wh = wh.view(batch, K, 2)
    wh[wh < 0] = 0
    if wh.size(2) == 2 * cat:  # cat spec
        wh = wh.view(batch, K, -1, 2)
        cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
        wh = wh.gather(2, cats.long()).squeeze(2)  # B x K x 2
    else:
        pass
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    ret['bboxes'] = bboxes

    regression_heads = ['tracking', 'dep', 'rot', 'dim', 'amodel_offset',
                        'nuscenes_att', 'velocity']

    for head in regression_heads:
        if head in output:
            ret[head] = _tranpose_and_gather_feat(
                output[head], inds).view(batch, K, -1)

    return ret


def generic_post_process(dets, c, s, h, w, num_classes, calibs=None, height=-1, width=-1):
    if not ('scores' in dets):
        return [{}], [{}]
    ret = []

    for i in range(len(dets['scores'])):
        preds = []
        trans = get_affine_transform(
            c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
        for j in range(len(dets['scores'][i])):
            if dets['scores'][i][j] < 0.3:  # opt.out_thresh:
                break

            item = {}

            item['score'] = dets['scores'][i][j]
            item['class'] = int(dets['clses'][i][j]) + 1
            item['ct'] = transform_preds_with_trans(
                (dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)

            if 'tracking' in dets:
                tracking = transform_preds_with_trans(
                    (dets['tracking'][i][j] + dets['cts'][i][j]).reshape(1, 2),
                    trans).reshape(2)
                item['tracking'] = tracking - item['ct']

            if 'bboxes' in dets:
                bbox = transform_preds_with_trans(
                    dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
                item['bbox'] = bbox

            if 'hps' in dets:
                pts = transform_preds_with_trans(
                    dets['hps'][i][j].reshape(-1, 2), trans).reshape(-1)
                item['hps'] = pts

            if 'dep' in dets and len(dets['dep'][i]) > j:
                item['dep'] = dets['dep'][i][j]

            if 'dim' in dets and len(dets['dim'][i]) > j:
                item['dim'] = dets['dim'][i][j]

            if 'rot' in dets and len(dets['rot'][i]) > j:
                item['alpha'] = get_alpha(dets['rot'][i][j:j + 1])[0]

            if 'rot' in dets and 'dep' in dets and 'dim' in dets \
                    and len(dets['dep'][i]) > j:
                if 'amodel_offset' in dets and len(dets['amodel_offset'][i]) > j:
                    ct_output = dets['bboxes'][i][j].reshape(2, 2).mean(axis=0)
                    amodel_ct_output = ct_output + dets['amodel_offset'][i][j]
                    ct = transform_preds_with_trans(
                        amodel_ct_output.reshape(1, 2), trans).reshape(2).tolist()
                else:
                    bbox = item['bbox']
                    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                item['ct'] = ct
                item['loc'], item['rot_y'] = ddd2locrot(
                    ct, item['alpha'], item['dim'], item['dep'], calibs[i])

            preds.append(item)

        ret.append(preds)

    return ret
