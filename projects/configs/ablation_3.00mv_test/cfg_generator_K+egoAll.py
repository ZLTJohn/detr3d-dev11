import mmengine
from mmengine.config import Config, DictAction
import shutil
from subprocess import PIPE
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tempfile
def save_hist(H, filename='NLstat'):
    # H = np.array([[1, 2, 3, 4],
    #             [5, 6, 7, 8],
    #             [9, 10, 11, 12],
    #             [13, 14, 15, 16]])  # added some commas and array creation code

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title(filename) 
    plt.imshow(H[::-1,:])
    ax.set_aspect('equal')
    ax.set_xlabel('ego frontal shift wrt. rare axle')
    ax.set_ylabel('ego height wrt. the ground')
    ax.set_xticks(range(len(Xs)),Xs)
    ax.set_yticks(range(len(Zs)),Zs[::-1])

    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    # cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.savefig(fname='debug/'+filename)
    
Zs = np.array(list(range(-15,16,5))) / 10.0
Xs = np.array(list(range(-45,16,5))) / 10.0
# Zs = np.array(list(range(0,2,3))) / 10.0
# Xs = np.array(list(range(-3,3,3))) / 10.0
waymo_brief = np.zeros((10,Zs.shape[0],Xs.shape[0]))
waymo_detail = np.array(waymo_brief,dtype=str)
for i,z in enumerate(Zs):
    for j,x in enumerate(Xs):
        work_dir = './work_dirs_3.00mv_ablate/ANK_K+egoAll_eval_on_LKK360_z{}_x{}'.format(z,x)
        src_cfg = 'projects/configs/ablation_3.00mv_test/K+egoAll_node47.py'
        ckpt = 'work_dirs_ablate/K+egoAll/epoch_24.pth'
        NLmeta = dict(
            lyft_egoZ = [dict(type='ego_transform'), dict(type='EgoTranslate', trans = [x,0,z])],
            kitti_egoZ = [dict(type='EgoTranslate', trans = [x,0,z])],
            K360_egoZ = [dict(type='EgoTranslate', trans = [x,0,z])],
            work_dir = work_dir
        )
        cfg = Config(NLmeta)
        tmp_dir = tempfile.TemporaryDirectory()
        Config.dump(cfg,tmp_dir.name+'/temp.py')
        with open(src_cfg,'r') as f:
            s = f.readlines()
            with open(tmp_dir.name+'/temp.py','a') as f1:
                f1.writelines(s)
        # run
        # child = subprocess.Popen(['tools/dist_test.sh',tmp_dir.name+'/temp.py',ckpt ,'8'])
        # child.wait()
        tmp_dir.cleanup()

        # get results
        with open(work_dir+'/brief_metric.txt','r') as result_file:
            lines = result_file.readlines()
            for k in range(2,8):
                items = lines[k].strip('\n').split(' ')
                waymo_brief[k,i,j] = items[1][:-1]
                waymo_detail[k,i,j] = items[2]
for k,name in zip(range(2,8),['A','K','K360','L','N','W']):
    save_hist(waymo_brief[k], filename='egoXZ_K_mv_'+name)
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