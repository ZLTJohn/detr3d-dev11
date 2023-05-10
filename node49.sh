# bash tools/dist_train.sh projects/configs/mono_ablation/1.00L_fx2070.py 8
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00N_fx2070.py 8
bash tools/dist_test.sh  /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_egotest/11.py work_dirs_mono_ablate/1.00L_fx2070/epoch_24.pth 8
bash tools/dist_test.sh  /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_egotest/12.py work_dirs_mono_ablate/1.00L_fx2070/epoch_24.pth 8
bash tools/dist_test.sh  /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_egotest/13.py work_dirs_mono_ablate/1.00L_fx2070/epoch_24.pth 8
bash tools/dist_test.sh  /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_egotest/14.py work_dirs_mono_ablate/1.00L_fx2070/epoch_24.pth 8
bash tools/dist_test.sh  /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_egotest/15.py work_dirs_mono_ablate/1.00L_fx2070/epoch_24.pth 8