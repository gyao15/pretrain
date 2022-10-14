import mmcv
import numpy as np
import os
from os import path as osp
from scipy.spatial.transform import Rotation
from mmdet3d.core.bbox import box_np_ops
from scipy import io as scio

dataset_name = 'once'
data_root = './data/once'


if dataset_name == 'once':
    CLASS_NAMES = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
    splits = ['train', 'val']
    for split in splits:
        split_dir = osp.join(data_root, 'ImageSets', '%s.txt' % split)
        info_path = './data/once/once_infos_%s.pkl' % split
        seq_list = [x.strip() for x in open(split_dir).readlines()]
        frame_num = {x: 0 for x in seq_list}
        infos = mmcv.load(info_path)
        for info in infos:
            seq_id = info['seq_id']
            lidar_path = info['lidar_path']
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            seq_path = osp.join(data_root, 'visual_data', seq_id)
            if not os.path.exists(seq_path):
                os.mkdir(seq_path)
            lidar_file = os.path.join(seq_path, 'lidar_roof')
            if not os.path.exists(lidar_file):
                os.mkdir(lidar_file)
            save_path = osp.join(lidar_file, '%06d.bin' % frame_num[seq_id])
            points.tofile(save_path)
            boxes = info['gt_boxes']
            names = info['gt_names']
            corners = box_np_ops.center_to_corner_box3d(boxes[:, :3], boxes[:, 3:6], boxes[:, -1],
                                                        origin=(0.5,0.5,0.5), axis=2)
            labels = np.array([CLASS_NAMES.index(name) for name in names])
            labels_file = os.path.join(seq_path, 'labels')
            if not os.path.exists(labels_file):
                os.mkdir(labels_file)
            save_path = osp.join(labels_file, '%06d.mat' % frame_num[seq_id])
            scio.savemat(save_path, {'boxes': corners, 'labels': labels})
            frame_num[seq_id] += 1

