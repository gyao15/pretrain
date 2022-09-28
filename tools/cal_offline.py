
import mmcv
import argparse
import copy
import os
import numpy as np
from mmcv import Config, DictAction
from os import path as osp
from mmdet3d.datasets import once_eval_utils
import scipy.io as scio
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='configs/once_post.py', help='config file path')
    parser.add_argument('--val', action='store_true', help='training or validation set')
    parser.add_argument('--dataset', type=str, default='once')
    args = parser.parse_args()
    return args

def once_error_analysis(results, classes, iou_thresh_dict):
    num_pos = np.zeros((8, 5))
    num_fp = np.zeros((8, 5))
    num_inacc = np.zeros((8, 5))
    num_miss = np.zeros((8, 5))
    
    for result in results:
        boxes = result['boxes_3d']
        dist = np.linalg.norm(boxes[:, :2], axis=1)
        iou = result['iou']
        score = result['score']
        name = result['name']
        dist = dist // 10
        dist[dist>7] = 7
        dist = dist.astype(np.int32)
        for d, i, s, n in zip(dist, iou, score, name):
            if s < 0:
                num_miss[d, classes.index(n)] += 1
            elif i >= iou_thresh_dict[n]:
                num_pos[d, classes.index(n)] += 1
            elif i > 0:
                num_inacc[d, classes.index(n)] += 1
            elif s > 0.1:
                num_fp[d, classes.index(n)] += 1
    print('Positive')
    print('distance: %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  >=%dm  total' % (10, 20, 30, 40, 50, 60, 70, 70))
    for i, c in enumerate(classes):
        print('%s:    %d   %d   %d   %d   %d   %d   %d   %d   %d' % (c, num_pos[0, i], num_pos[1, i], num_pos[2, i], num_pos[3, i], num_pos[4, i], num_pos[5, i], num_pos[6, i], num_pos[7, i], num_pos[:, i].sum()))
    print('Inaccurate')
    print('distance: %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  >=%dm  total' % (10, 20, 30, 40, 50, 60, 70, 70))
    for i, c in enumerate(classes):
        print('%s:    %d   %d   %d   %d   %d   %d   %d   %d   %d' % (c, num_inacc[0, i], num_inacc[1, i], num_inacc[2, i], num_inacc[3, i], num_inacc[4, i], num_inacc[5, i], num_inacc[6, i], num_inacc[7, i], num_inacc[:, i].sum()))
    print('False Alarm')
    print('distance: %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  >=%dm  total' % (10, 20, 30, 40, 50, 60, 70, 70))
    for i, c in enumerate(classes):
        print('%s:    %d   %d   %d   %d   %d   %d   %d   %d   %d' % (c, num_fp[0, i], num_fp[1, i], num_fp[2, i], num_fp[3, i], num_fp[4, i], num_fp[5, i], num_fp[6, i], num_fp[7, i], num_fp[:, i].sum()))
    print('Missing')
    print('distance: %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  %dm<  >=%dm  total' % (10, 20, 30, 40, 50, 60, 70, 70))
    for i, c in enumerate(classes):
        print('%s:    %d   %d   %d   %d   %d   %d   %d   %d   %d' % (c, num_miss[0, i], num_miss[1, i], num_miss[2, i], num_miss[3, i], num_miss[4, i], num_miss[5, i], num_miss[6, i], num_miss[7, i], num_miss[:, i].sum()))

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    info_path = cfg.val_info_path if args.val else cfg.train_info_path
    result_path = cfg.val_result_path if args.val else cfg.train_result_path
    save_path = cfg.val_save_path if args.val else cfg.train_save_path
    infos = mmcv.load(info_path)
    results = mmcv.load(result_path)
    if args.dataset == 'once':
        iou_threshold_dict = {
            'Car': 0.7,
            'Bus': 0.7,
            'Truck': 0.7,
            'Pedestrian': 0.3,
            'Cyclist': 0.5
        }
        gt_annos = []
        for info in infos:
            gt_boxes = info['gt_boxes']
            gt_boxes = info['gt_boxes']
            gt_boxes[:, [3,4]] = gt_boxes[:, [4,3]]
            gt_boxes[:, -1] = 3 * np.pi/2 - gt_boxes[:, -1]
            gt_anno = {'name': info['gt_names'], 'boxes_3d': gt_boxes, 'frame_id': info['frame_id']}
            gt_annos.append(gt_anno)

        assert len(results) == len(gt_annos)

        num_samples = len(gt_annos)
        num_parts = 100
        split_parts = once_eval_utils.compute_split_parts(num_samples, num_parts)
        ious = once_eval_utils.compute_iou3d(gt_annos, results, split_parts, with_heading=True)
        classes = ['Car', 'Truck', 'Bus', 'Cyclist', 'Pedestrian']
        final_list = []
        for sample_idx in mmcv.track_iter_progress(range(num_samples)):
            gt_anno = gt_annos[sample_idx]
            pred_anno = results[sample_idx]
            iou = ious[sample_idx]
            pred_box3d = pred_anno['boxes_3d']
            score = pred_anno['score']
            pred_name = pred_anno['name']
            
            gt_name = gt_anno['name']
            num_gt = iou.shape[0]
            num_pred = iou.shape[1]
            missing_gt_idx = np.ones((num_gt,))
            pred_iou = []
            for j in range(num_pred):
                max_iou = -1
                max_idx = -1
                for i in range(num_gt):
                    if pred_name[j] != gt_name[i]:
                        continue
                    if iou[i, j] > max_iou:
                        max_iou = iou[i, j]
                        max_idx = i
                if max_iou > iou_threshold_dict[pred_name[j]]:
                    missing_gt_idx[max_idx] = 0
                pred_iou.append(max_iou)
            pred_iou = np.array(pred_iou)
            missing_gt_name = gt_name[missing_gt_idx>0]
            if len(missing_gt_name) > 0:
                missing_gt_box3d = gt_anno['boxes_3d'][missing_gt_idx>0]
                missing_gt_iou = np.ones((len(missing_gt_name),))
                missing_gt_score = np.ones((len(missing_gt_name),)) * (-1)
                pred_name = np.concatenate([pred_name, missing_gt_name])
                pred_box3d = np.concatenate([pred_box3d, missing_gt_box3d], axis=0)
                pred_iou = np.concatenate([pred_iou, missing_gt_iou])
                score = np.concatenate([score, missing_gt_score])
            #print(sample_idx, len(score), len(pred_score), len(pred_name), num_gt, num_pred, len(missing_gt_name))
            
            final_list.append(
                {
                    'frame_id': pred_anno['frame_id'],
                    'boxes_3d': pred_box3d,
                    'name': pred_name,
                    'score': score,
                    'iou': pred_iou
                }
            )
        mmcv.dump(final_list, save_path)
        once_error_analysis(final_list, classes, iou_threshold_dict)

def error():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    save_path = cfg.val_save_path if args.val else cfg.train_save_path
    infos = mmcv.load(save_path)
    classes = ['Car', 'Truck', 'Bus', 'Cyclist', 'Pedestrian']
    iou_threshold_dict = {
        'Car': 0.7,
        'Bus': 0.7,
        'Truck': 0.7,
        'Pedestrian': 0.3,
        'Cyclist': 0.5
    }
    once_error_analysis(infos, classes, iou_threshold_dict)

def to_mat():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    result_path = cfg.val_result_path if args.val else cfg.train_result_path
    save_path = cfg.val_save_path if args.val else cfg.train_save_path
    infos = mmcv.load(save_path)
    results = mmcv.load(result_path)
    out = {'frame_id': [], 'boxes_3d': [], 'name': [], 'labels': [], 'score': [], 'iou': [], 'features': [], 'box_features': []}
    classes = ['Car', 'Truck', 'Bus', 'Cyclist', 'Pedestrian']
    if args.dataset == 'once':
        for info, result in zip(infos, results):
            for k in info.keys():
                if k != 'frame_id':
                    out[k].append(info[k])

            #print(np.ones((len(info['name']),)), info['frame_id'])
            out['frame_id'].append(np.ones((len(info['name']),)) * int(info['frame_id']))
            out['labels'].append(np.zeros((len(info['name']),)))
            for i in range(len(info['name'])):
                out['labels'][-1][i] = classes.index(out['name'][-1][i])
            out['features'].append(result['features'])
            out['box_features'].append(result['box_features'])
        
        for k, v in out.items():
            out[k] = np.concatenate(v, axis=0)
    
    scio.savemat(save_path.replace('pkl', 'mat'), out)

def get_sim():
    cfg = Config.fromfile('configs/once_post.py')
    result_path = cfg.val_result_path
    save_path = cfg.val_save_path
    infos = mmcv.load(save_path)
    results = mmcv.load(result_path)
    out = {'frame_id': [], 'boxes_3d': [], 'name': [], 'score': [], 'iou': [], 'features': [], 'box_features': []}
    for info, result in zip(infos, results):
        for k in info.keys():
            if k != 'frame_id':
                out[k].append(info[k])

        #print(np.ones((len(info['name']),)), info['frame_id'])
        out['frame_id'].append(np.ones((len(info['name']),)) * int(info['frame_id']))
        out['features'].append(result['features'].astype(np.float32))
        out['box_features'].append(result['box_features'].astype(np.float32))
    
    for k, v in out.items():
        out[k] = np.concatenate(v, axis=0)

    result_path = cfg.train_result_path
    save_path = cfg.train_save_path
    infos = mmcv.load(save_path)
    results = mmcv.load(result_path)
    train_out = {'frame_id': [], 'boxes_3d': [], 'name': [], 'score': [], 'iou': [], 'features': [], 'box_features': []}
    for info, result in zip(infos, results):
        for k in info.keys():
            if k != 'frame_id':
                train_out[k].append(info[k])

        #print(np.ones((len(info['name']),)), info['frame_id'])
        train_out['frame_id'].append(np.ones((len(info['name']),)) * int(info['frame_id']))
        train_out['features'].append(result['features'].astype(np.float32))
        train_out['box_features'].append(result['box_features'].astype(np.float32))
    
    for k, v in train_out.items():
        train_out[k] = np.concatenate(v, axis=0)

    mask = out['score'] >= 0
    train_mask = train_out['score'] >= 0
    for k in ['frame_id', 'boxes_3d', 'name', 'score', 'iou']:
        out[k] = out[k][mask]
        train_out[k] = train_out[k][train_mask]
    
    out['dist'] = np.linalg.norm(out['boxes_3d'][:,:2], axis=1)
    
    val_out = {'frame_id': [], 'boxes_3d': [], 'name': [], 'score': [], 'iou': [], 'features': [], 'box_features': []}
    for c in cfg.class_names:
        for i in range(1, len(cfg.dist_intervals)):
            for j in range(1, len(cfg.iou_intervals)):
                mask = (out['name'] == c) & (out['dist'] > cfg.dist_intervals[i-1]) & (out['dist'] <= cfg.dist_intervals[i]) & (out['iou'] > cfg.iou_intervals[j-1]) & (out['iou'] <= cfg.iou_intervals[j])
                indices = np.where(mask==True)[0]
                if len(indices) > 50:
                    indices = np.random.choice(indices, size=50, replace=False)
                for k in val_out.keys():
                    val_out[k].append(out[k][indices])
    for k, v in val_out.items():
        val_out[k] = np.concatenate(v, axis=0)

    sim = np.zeros((val_out['features'].shape[0], train_out['features'].shape[0]))
    for i, feat in tqdm(enumerate(val_out['features'])):
        sim[i] = np.dot(train_out['features'], feat) / np.linalg.norm(feat) / np.linalg.norm(train_out['features'], axis=1)

    train_out.pop('features')
    train_out.pop('box_features')

    mmcv.dump([train_out, val_out], 'work_dirs/transfusion_once_voxel_L/sim.pkl')
    np.save('work_dirs/transfusion_once_voxel_L/sim.npy', sim)


def get_mv():
    cfg = Config.fromfile('configs/once_post.py')
    result_path = cfg.val_result_path
    save_path = cfg.val_save_path
    infos = mmcv.load(save_path)
    results = mmcv.load(result_path)
    out = {'frame_id': [], 'boxes_3d': [], 'name': [], 'score': [], 'iou': [], 'features': [], 'box_features': []}
    for info, result in zip(infos, results):
        for k in info.keys():
            if k != 'frame_id':
                out[k].append(info[k])

        #print(np.ones((len(info['name']),)), info['frame_id'])
        out['frame_id'].append(np.ones((len(info['name']),)) * int(info['frame_id']))
        out['features'].append(result['features'].astype(np.float32))
        out['box_features'].append(result['box_features'].astype(np.float32))
    
    for k, v in out.items():
        out[k] = np.concatenate(v, axis=0)

    result_path = cfg.train_result_path
    save_path = cfg.train_save_path
    infos = mmcv.load(save_path)
    results = mmcv.load(result_path)
    train_out = {'frame_id': [], 'boxes_3d': [], 'name': [], 'score': [], 'iou': [], 'features': [], 'box_features': []}
    for info, result in zip(infos, results):
        for k in info.keys():
            if k != 'frame_id':
                train_out[k].append(info[k])

        #print(np.ones((len(info['name']),)), info['frame_id'])
        train_out['frame_id'].append(np.ones((len(info['name']),)) * int(info['frame_id']))
        train_out['features'].append(result['features'].astype(np.float32))
        train_out['box_features'].append(result['box_features'].astype(np.float32))
    
    for k, v in train_out.items():
        train_out[k] = np.concatenate(v, axis=0)

    mask = out['score'] >= 0
    train_mask = train_out['score'] >= 0
    for k in ['frame_id', 'boxes_3d', 'name', 'score', 'iou']:
        out[k] = out[k][mask]
        train_out[k] = train_out[k][train_mask]
    
    out['dist'] = np.linalg.norm(out['boxes_3d'][:,:2], axis=1)
    val_mv = {}
    for c in cfg.class_names:
        val_mv[c] = {}
        mask = (out['name'] == c) & (out['dist'] >= 0) & (out['dist'] < 30)
        if c in ['Car', 'Bus', 'Truck']:
            pos_mask = mask & (out['iou'] >= 0.7)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.7)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['0_pos'] = {}
            val_mv[c]['0_inacc'] = {}
            val_mv[c]['0_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['0_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['0_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['0_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        elif c == 'Cyclist':
            pos_mask = mask & (out['iou'] >= 0.5)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.5)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['0_pos'] = {}
            val_mv[c]['0_inacc'] = {}
            val_mv[c]['0_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['0_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['0_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['0_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        else:
            pos_mask = mask & (out['iou'] >= 0.3)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.3)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['0_pos'] = {}
            val_mv[c]['0_inacc'] = {}
            val_mv[c]['0_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['0_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['0_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['0_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        mask = (out['name'] == c) & (out['dist'] >= 30) & (out['dist'] < 50)
        if c in ['Car', 'Bus', 'Truck']:
            pos_mask = mask & (out['iou'] >= 0.7)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.7)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['30_pos'] = {}
            val_mv[c]['30_inacc'] = {}
            val_mv[c]['30_neg'] = {}
            for k in ['iou', 'score', 'features', 'box_features']:
                val_mv[c]['30_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['30_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['30_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        elif c == 'Cyclist':
            pos_mask = mask & (out['iou'] >= 0.5)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.5)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['30_pos'] = {}
            val_mv[c]['30_inacc'] = {}
            val_mv[c]['30_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['30_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['30_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['30_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        else:
            pos_mask = mask & (out['iou'] >= 0.3)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.3)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['30_pos'] = {}
            val_mv[c]['30_inacc'] = {}
            val_mv[c]['30_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['30_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['30_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['30_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        mask = (out['name'] == c) & (out['dist'] >= 50)
        if c in ['Car', 'Bus', 'Truck']:
            pos_mask = mask & (out['iou'] >= 0.7)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.7)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['50_pos'] = {}
            val_mv[c]['50_inacc'] = {}
            val_mv[c]['50_neg'] = {}
            for k in ['iou', 'score', 'features', 'box_features']:
                val_mv[c]['50_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['50_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['50_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        elif c == 'Cyclist':
            pos_mask = mask & (out['iou'] >= 0.5)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.5)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['50_pos'] = {}
            val_mv[c]['50_inacc'] = {}
            val_mv[c]['50_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['50_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['50_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['50_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        else:
            pos_mask = mask & (out['iou'] >= 0.3)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.3)
            neg_mask = mask & (out['iou'] <= 0)
            val_mv[c]['50_pos'] = {}
            val_mv[c]['50_inacc'] = {}
            val_mv[c]['50_neg'] = {}
            for k in ['features', 'box_features']:
                val_mv[c]['50_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                val_mv[c]['50_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                val_mv[c]['50_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])

    train_out['dist'] = np.linalg.norm(train_out['boxes_3d'][:,:2], axis=1)
    train_mv = {}
    for c in cfg.class_names:
        train_mv[c] = {}
        mask = (out['name'] == c) & (out['dist'] >= 0) & (out['dist'] < 30)
        if c in ['Car', 'Bus', 'Truck']:
            pos_mask = mask & (out['iou'] >= 0.7)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.7)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['0_pos'] = {}
            train_mv[c]['0_inacc'] = {}
            train_mv[c]['0_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['0_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['0_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['0_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        elif c == 'Cyclist':
            pos_mask = mask & (out['iou'] >= 0.5)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.5)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['0_pos'] = {}
            train_mv[c]['0_inacc'] = {}
            train_mv[c]['0_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['0_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['0_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['0_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        else:
            pos_mask = mask & (out['iou'] >= 0.3)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.3)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['0_pos'] = {}
            train_mv[c]['0_inacc'] = {}
            train_mv[c]['0_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['0_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['0_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['0_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        mask = (out['name'] == c) & (out['dist'] >= 30) & (out['dist'] < 50)
        if c in ['Car', 'Bus', 'Truck']:
            pos_mask = mask & (out['iou'] >= 0.7)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.7)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['30_pos'] = {}
            train_mv[c]['30_inacc'] = {}
            train_mv[c]['30_neg'] = {}
            for k in ['iou', 'score', 'features', 'box_features']:
                train_mv[c]['30_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['30_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['30_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        elif c == 'Cyclist':
            pos_mask = mask & (out['iou'] >= 0.5)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.5)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['30_pos'] = {}
            train_mv[c]['30_inacc'] = {}
            train_mv[c]['30_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['30_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['30_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['30_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        else:
            pos_mask = mask & (out['iou'] >= 0.3)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.3)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['30_pos'] = {}
            train_mv[c]['30_inacc'] = {}
            train_mv[c]['30_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['30_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['30_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['30_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        mask = (out['name'] == c) & (out['dist'] >= 50)
        if c in ['Car', 'Bus', 'Truck']:
            pos_mask = mask & (out['iou'] >= 0.7)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.7)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['50_pos'] = {}
            train_mv[c]['50_inacc'] = {}
            train_mv[c]['50_neg'] = {}
            for k in ['iou', 'score', 'features', 'box_features']:
                train_mv[c]['50_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['50_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['50_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        elif c == 'Cyclist':
            pos_mask = mask & (out['iou'] >= 0.5)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.5)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['50_pos'] = {}
            train_mv[c]['50_inacc'] = {}
            train_mv[c]['50_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['50_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['50_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['50_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
        else:
            pos_mask = mask & (out['iou'] >= 0.3)
            inacc_mask = mask & (out['iou'] > 0) & (out['iou'] < 0.3)
            neg_mask = mask & (out['iou'] <= 0)
            train_mv[c]['50_pos'] = {}
            train_mv[c]['50_inacc'] = {}
            train_mv[c]['50_neg'] = {}
            for k in ['features', 'box_features']:
                train_mv[c]['50_pos'][k] = np.sum(out[k][pos_mask] * out['score'][pos_mask, np.newaxis], axis=0) / np.sum(out['score'][pos_mask])
                train_mv[c]['50_inacc'][k] = np.sum(out[k][inacc_mask] * out['score'][inacc_mask, np.newaxis], axis=0) / np.sum(out['score'][inacc_mask])
                train_mv[c]['50_neg'][k] = np.sum(out[k][neg_mask] * out['score'][neg_mask, np.newaxis], axis=0) / np.sum(out['score'][neg_mask])
    
    mmcv.dump(val_mv, 'work_dirs/transfusion_once_voxel_L/val_mv.json')
    mmcv.dump(train_mv, 'work_dirs/transfusion_once_voxel_L/train_mv.json')
        

if __name__ == '__main__':
    #main()
    #to_mat()
    #get_sim()
    #get_mv()
    error()


