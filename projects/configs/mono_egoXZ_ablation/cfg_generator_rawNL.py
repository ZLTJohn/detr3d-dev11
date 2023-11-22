import mmengine
from mmengine.config import Config, DictAction
import shutil
from subprocess import PIPE
import subprocess
import numpy as np
import matplotlib.pyplot as plt
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

Zs = np.array(list(range(-33,2,3))) / 10.0
Xs = np.array(list(range(-45,16,5))) / 10.0
# Zs = np.array(list(range(0,2,3))) / 10.0
# Xs = np.array(list(range(-3,3,3))) / 10.0
nusc_brief = np.zeros((Zs.shape[0],Xs.shape[0]))
nusc_detail = np.array(nusc_brief,dtype=str)
lyft_brief = np.zeros((Zs.shape[0],Xs.shape[0]))
lyft_detail = np.array(lyft_brief,dtype=str)
for i,z in enumerate(Zs):
    for j,x in enumerate(Xs):
        work_dir = './work_dirs_mono_egoXZ_ablate/W_fx2070_LidarCoordNL_z{}_x{}'.format(z,x)
        NLmeta = dict(
            nusc_egoZ = [dict(type='EgoTranslate', trans = [x,0,z])],
            lyft_egoZ = [dict(type='EgoTranslate', trans = [x,0,z])],
            work_dir = work_dir
        )
        cfg = Config(NLmeta)
        Config.dump(cfg,'debug/temp1.py')
        with open('projects/configs/mono_egoXZ_ablation/W_fx2070_eval_LidarCoordNL.py','r') as f:
            s = f.readlines()
            with open('debug/temp1.py','a') as f1:
                f1.writelines(s)
        # run
        # child = subprocess.Popen(['tools/dist_test.sh','debug/temp1.py','work_dirs_extended/1.00W_front_r50_fullres/epoch_24.pth' ,'8'])
        # child.wait()
        # get results
        with open(work_dir+'/brief_metric.txt','r') as result_file:
            lines = result_file.readlines()
            
            items = lines[1].strip('\n').split(' ')
            lyft_brief[i,j] = items[1][:-1]
            lyft_detail[i,j] = items[2]
            
            items = lines[2].strip('\n').split(' ')
            nusc_brief[i,j] = items[1][:-1]
            nusc_detail[i,j] = items[2]

print(nusc_brief/100)
print(lyft_brief/100)
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