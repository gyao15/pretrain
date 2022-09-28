
import mmcv
import argparse
import copy
import os
import numpy as np
from mmcv import Config, DictAction
from os import path as osp
from mmdet3d.datasets import once_eval_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
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
                pred_score = np.concatenate([score, missing_gt_score])
            
            final_list.append(
                {
                    'frame_id': pred_anno['frame_id'],
                    'boxes_3d': pred_box3d,
                    'name': pred_name,
                    'score': pred_score,
                    'iou': pred_iou
                }
            )
        mmcv.dump(final_list, save_path)
        once_error_analysis(final_list, classes, iou_threshold_dict)

if __name__ == '__main__':
    main()


