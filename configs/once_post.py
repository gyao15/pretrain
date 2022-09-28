from cmath import inf


train_info_path = 'data/once/once_infos_train.pkl'
val_info_path = 'data/once/once_infos_val.pkl'
train_result_path = 'work_dirs/transfusion_once_voxel_L/train_results/results_once.pkl'
val_result_path = 'work_dirs/transfusion_once_voxel_L/eval_results/results_once.pkl'
train_save_path = 'work_dirs/transfusion_once_voxel_L/train_results/results.pkl'
val_save_path = 'work_dirs/transfusion_once_voxel_L/eval_results/results.pkl'

dist_intervals = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
iou_intervals = [-2.0, 0.0, 0.3, 0.5, 0.7, 1.0]
class_names = [
    'Car', 'Truck', 'Bus', 'Cyclist', 'Pedestrian'
]
enlarge_ratios = [1.0, 2.0, 3.0]
