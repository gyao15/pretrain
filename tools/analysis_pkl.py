import mmcv

work_dir = '/data1/dn/backup/TransFusion/work_dirs/220921transfusion_once_voxel_l/results.pkl'
data = mmcv.load(work_dir)
print(type(data))
print(len(data))
# for key, value in data[0][0]:
#     print(key)