import numpy as np
metric_path = '/home/zhenglt/mmdev11/detr3d-dev11/work_dirs_bevdet/W_r50_mono_bs8_smallres+littleimgaug/'
log_file = '20230524_131443/20230524_131443.log'
f = open(metric_path+log_file,'r')
S = f.read()
L = 0
R = 0
while True:
    L = S.find('waymo: ',R+1)
    if L==-1:
        break
    R = S.find('\n',L+1)
    print(S[L:R])
