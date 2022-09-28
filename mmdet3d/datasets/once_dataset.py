import mmcv
import numpy as np
import pickle
import copy
from PIL import Image
import tempfile
from os import path as osp
from .once_toolkits import Octopus
import torch

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .once_eval_utils import get_evaluation_results

@DATASETS.register_module()
class OnceDataset(Custom3DDataset):
    CLASSES = ('Car', 'Bus', 'Truck', 'Cyclist', 'Pedestrian')

    def __init__(self, ann_file, num_views=6, pipeline=None, data_root=None, classes=None, modality=None,
                 box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False, use_valid_flag=False) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        self.use_valid_flag = use_valid_flag
        self.num_views = num_views
        assert self.num_views <= 7
        self.cam_names = np.array(['cam01', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09', 'cam03'])
        self.cam_names = self.cam_names[:self.num_views]
        self.cam_tags = ['top', 'left_back', 'left_front', 'right_front', 'right_back', 'back', 'top2']
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data_infos = mmcv.load(ann_file)
        data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx = info['frame_id'],
            seq_id = info['seq_id'],
            frame_id = info['frame_id'],
            pts_filename= info['lidar_path'],
            timestamp=float(info['timestamp']) / 1e3,
        )
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            for cam_name in self.cam_names:
                cam_info = info['cams'][cam_name]
                image_paths.append(cam_info['data_path'])
                img2lidar_rt = cam_info['cam2velo']
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:3, :3] = intrinsic
                lidar2img_rt = (viewpad @ np.linalg.inv(img2lidar_rt))
                lidar2img_rts.append(torch.from_numpy(lidar2img_rt))
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=torch.stack(lidar2img_rts, dim=0),
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0

        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        #gt_bboxes_2d = info['gt_boxes_2d'][mask]
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                print(cat)
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            #gt_bboxes_2d=gt_bboxes_2d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, out_path=None):
        once_annos = []
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            frame_id = self.data_infos[sample_id]['frame_id']
            pred_dict = output_to_once_box(det['pts_bbox'], mapped_class_names)
            pred_dict['frame_id'] = frame_id
            once_annos.append(pred_dict)
        if out_path is not None: 
            mmcv.mkdir_or_exist(out_path)
            res_path = osp.join(out_path, 'results_once.pkl')
            mmcv.dump(once_annos, res_path)
        return once_annos

    def evaluate(self, results, metric='bbox', out_dir=None):
        gt_annos = []
        for info in self.data_infos:
            gt_boxes = info['gt_boxes']
            gt_boxes[:, [3,4]] = gt_boxes[:, [4,3]]
            gt_boxes[:, -1] = 3 * np.pi/2 - gt_boxes[:, -1]
            gt_anno = {'name': info['gt_names'], 'boxes_3d': gt_boxes}
            gt_annos.append(gt_anno)
        det_annos = self._format_bbox(results, out_dir)
        ap_result_str, ap_dict = get_evaluation_results(gt_annos, det_annos, self.CLASSES)
        if out_dir is not None:
            ap_path = osp.join(out_dir, 'eval_once.json')
            mmcv.dump(ap_dict, ap_path)
        return ap_result_str

def output_to_once_box(detection, mapped_class_names, score_thr=0.001):
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    
    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    box_yaw = 3 * np.pi/2 - box_yaw
    box_dims = box_dims[:, [1, 0, 2]]
    boxes = np.concatenate([box_gravity_center, box_dims, box_yaw[:, np.newaxis]], axis=-1)
    mask = scores > score_thr
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    

    if len(scores) == 0:
        pred_dict = {
            'name': np.zeros((0,)),
            'score': np.zeros((0,)),
            'boxes_3d': np.zeros((0,)),
        }
    else:
        pred_dict = {
            'name': np.array(mapped_class_names)[labels],
            'score': scores,
            'boxes_3d': boxes,
        }
    if 'features' in detection:
        features = detection['features'].numpy()
        pred_dict['features'] = features[mask]
    if 'box_features' in detection:
        box_features = detection['box_features'].numpy()
        pred_dict['box_features'] = box_features[mask]
    return pred_dict
        

