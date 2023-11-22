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
    
Zs = np.array(list(range(-15,16,5))) / 10.0
Xs = np.array(list(range(-45,16,5))) / 10.0
# Zs = np.array(list(range(0,2,3))) / 10.0
# Xs = np.array(list(range(-3,3,3))) / 10.0
nusc_brief = np.zeros((Zs.shape[0],Xs.shape[0]))
nusc_detail = np.array(nusc_brief,dtype=str)
lyft_brief = np.zeros((Zs.shape[0],Xs.shape[0]))
lyft_detail = np.array(lyft_brief,dtype=str)
for i in range(1,25):
        # run
        child = subprocess.Popen(['tools/dist_test.sh','projects/configs/bevdet_ablation/1.00L.py','work_dirs_bevdet_mono_ablate/1.00L/epoch_{}.pth'.format(i) ,'8'])
        child.wait()
        # get results

            

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