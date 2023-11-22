import mmengine
from mmengine.config import Config, DictAction
import shutil
from subprocess import PIPE
import subprocess
import numpy as np
import tempfile
waymo_brief = []
for hfactor in [1.0,0.8,0.6,0.4]:
    for hsize in [853]:
        ckpt = '/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/0.33w_r50/epoch_24.pth'
        src_cfg = '/home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_W_crop_test/bevdet_r50_0.33w.py'
        h0 = (1280-hsize)*hfactor
        h1 = h0+hsize
        work_dir = './work_dirs_mono_crop_ablate/W_bevdet_imgh_{}_to_{}'.format(h0,h1)
        meta = dict(
            hfactor = hfactor,
            work_dir = work_dir,
            waymo_crop_wh=(864, 384),
            waymo_crop_hw=(384, 864),
        )
        cfg = Config(meta)
        tmp_dir = tempfile.TemporaryDirectory()
        Config.dump(cfg,tmp_dir.name+'/temp.py')
        with open(src_cfg,'r') as f:
            s = f.readlines()
            with open(tmp_dir.name+'/temp.py','a') as f1:
                f1.writelines(s)
        # run
        child = subprocess.Popen(['tools/dist_test.sh',tmp_dir.name+'/temp.py',ckpt ,'8'])
        # child = subprocess.Popen(['tools/dist_train.sh',tmp_dir.name+'/temp.py' ,'1'])
        child.wait()
        tmp_dir.cleanup()
        # get results
        with open(work_dir+'/brief_metric.txt','r') as result_file:
            lines = result_file.readlines()
            
            items = lines[1].strip('\n').split(' ')
            waymo_brief.append(items[1][:-1])

breakpoint()

# NLmeta = dict(
#     nusc_egoZ = [dict(type='ego_transform')],
#     lyft_egoZ = [dict(type='ego_transform')],
#     work_dir = './work_dirs_mono_egoXZ_ablate/NL_EGO'
# )
# cfg = Config(NLmeta)
# Config.dump(cfg,'debug.py')
# with open('W_fx2070_eval_NL.py','r') as f:
#     s = f.readlines()
#     with open('debug.py','a') as f1:
#         f1.writelines(s)