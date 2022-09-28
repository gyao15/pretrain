import mmcv
import os
from tqdm import tqdm

data_root = './data/once/data'
seq_ids = os.listdir(data_root)
seq_ids.sort()
infos = []

for seq_id in tqdm(seq_ids):
    seq_path = os.path.join(data_root, seq_id)
    cams = os.listdir(seq_path)
    cams.sort()

    for cam in cams:
        if 'cam' in cam:
            cam_path = os.path.join(seq_path, cam)
            img_name = os.listdir(cam_path)
            img_name.sort()
            info = [{'img_info': {'filename': os.path.join(cam_path, x)}} for x in img_name]
            infos += info

print(len(infos))
mmcv.dump(infos, os.path.join(data_root, 'pretrain_img_infos.pkl'))


