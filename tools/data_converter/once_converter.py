import mmcv
import numpy as np
import os
from collections import OrderedDict
from os import path as osp
from scipy.spatial.transform import Rotation
from mmdet3d.core.bbox import box_np_ops

def create_once_infos(root_path, info_prefix, splits=['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']):
    for split in splits:
        infos = _fill_split_infos(root_path, split)
        info_path = osp.join(root_path, '%s_infos_%s.pkl' % (info_prefix, split))
        mmcv.dump(infos, info_path)

def _fill_split_infos(data_root, split):
    split_dir = osp.join(data_root, 'ImageSets', '%s.txt' % split)
    seq_list = [x.strip() for x in open(split_dir).readlines()]
    infos = []
    for seq_id in seq_list:
        seq_infos = _fill_seq_infos(data_root, split, seq_id)
        infos.extend(seq_infos)
    return infos

def _fill_seq_infos(data_root, split, seq_id):
    print('%s seq_id: %s' % (split, seq_id))
    seq_infos = []
    seq_path = osp.join(data_root, 'data', seq_id)
    json_path = osp.join(seq_path, '%s.json' % seq_id)
    json_infos = mmcv.load(json_path)

    meta_info = json_infos['meta_info']
    calib = json_infos['calib']
    i = 0

    for frame in mmcv.track_iter_progress(json_infos['frames']):
        
        frame_id = frame['frame_id']
        lidar_path = osp.join(seq_path, 'lidar_roof', '%s.bin' % frame_id)
        mmcv.check_file_exist(lidar_path)

        pose = np.array(frame['pose'])
        pose_rotation = Rotation.from_quat(pose[:4]).as_matrix()
        pose_translation = pose[4:]

        info = {
            'lidar_path': lidar_path,
            'seq_id': seq_id,
            'frame_id': frame_id,
            'timestamp': int(frame_id),
            'meta_info': meta_info,
            'pose_rotation': pose_rotation,
            'pose_translation': pose_translation,
            'prev_id': None,
            'next_id': None,
            'cams': dict()
        }
        if i > 0:
            info['prev_id'] = json_infos['frames'][i-1]['frame_id']
        if i < len(json_infos['frames']) - 1:
            info['next_id'] = json_infos['frames'][i+1]['frame_id']

        camera_names = ['cam01', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09', 'cam03']
        for cam in camera_names:
            cam_path = osp.join(seq_path, cam, '%s.jpg' % frame_id)
            cam_info = {
                'data_path': cam_path,
                'cam2velo': np.array(calib[cam]['cam_to_velo']),
                'cam_intrinsic': np.array(calib[cam]['cam_intrinsic']),
                'distortion': np.array(calib[cam]['distortion'])
            }
            info['cams'].update({cam: cam_info})
        
        if 'annos' in frame:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            annos = frame['annos']
            boxes_3d = np.array(annos['boxes_3d'])
            
            if boxes_3d.shape[0] == 0:
                continue
            boxes_3d = boxes_3d[:, [0,1,2,4,3,5,6]]
            boxes_3d[:, -1] = 3 * np.pi/2 - boxes_3d[:, -1]
            boxes_2d = {cam: np.array(annos['boxes_2d'][cam]) for cam in camera_names}
            #rbbox_corners = box_np_ops.center_to_corner_box3d(boxes_3d[:, :3], boxes_3d[:, 3:6], boxes_3d[:, 6], origin=(0.5,0.5,0.5), axis=2)
            #surfaces = box_np_ops.corner_to_surfaces_3d(rbbox_corners)
            #print(rbbox_corners)
            indices = box_np_ops.points_in_rbbox(points[:, :3], boxes_3d, origin=(0.5,0.5,0.5))
            info['gt_boxes'] = boxes_3d
            info['gt_names'] = np.array(annos['names'])
            info['gt_boxes_2d'] = boxes_2d
            info['num_lidar_pts'] = indices.sum(0)
            info['valid_flag'] = info['num_lidar_pts'] > 0
            seq_infos.append(info)
        else:
            if (split == 'train') or (split == 'val'):
                continue
            else:
                seq_infos.append((info))
        i = i + 1

    return seq_infos
            
