import mmengine
from mmengine.config import Config, DictAction
import shutil
from subprocess import PIPE
import subprocess
import numpy as np
import tempfile
waymo_brief = []
for focal_length in [900,1200,1500,1780,2070,2400,2700,3000]:
    ckpt = '/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00W_fx2070_recheck/epoch_24.pth'
    src_cfg = '/home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_fx_ablation/1.00W_fx2070_recheck.py'
    work_dir = './work_dirs_mono_fx_ablation/1.00W_fx2070_evalA_fx{}'.format(focal_length)
    meta = dict(
        work_dir = work_dir,
        focal_length = focal_length
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