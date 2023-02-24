import pandas
import os.path as osp
import os
import shutil
import mmengine
# https://github.com/open-mmlab/mmdetection3d/pull/2178
# waymo data prefix bug: Left-side right
def copy_or_exist(src,dst):
    suffix = osp.basename(src)
    if not osp.exists(osp.dirname(dst)):
        os.makedirs(osp.dirname(dst))
    if not osp.exists(dst):
        shutil.copy(src,dst)
    return 1

def fix_one_pkl(pkl, new_prefix):
    print('fixing {}'.format(pkl))
    info = pandas.read_pickle(pkl)
    bar = mmengine.ProgressBar(len(info['data_list']))
    for i,item in enumerate(info['data_list']):
        old_prefix= list(item['images'].keys())
        img = item['images']
        tmp = []
        for key in old_prefix:
            tmp.append(img.pop(key))
        for i in range(len(new_prefix)):
            img[new_prefix[i]] = tmp[i]
        bar.update()
    print('rewriting pkl file {}...'.format(pkl))
    mmengine.dump(info, pkl)

newprefix = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_SIDE_LEFT','CAM_SIDE_RIGHT']
import glob
files=  glob.glob('/localdata_ssd/waymo_dev1x/*.pkl')
for file in files:
    fix_one_pkl(file,newprefix)
