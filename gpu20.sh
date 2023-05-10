# bash tools/dist_test.sh projects/configs/mono_ablation/1.00K360_fx3100.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00K360_fx3100/epoch_24.pth 8
# bash tools/dist_test.sh projects/configs/mono_ablation/1.00L_fx4140.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00L_fx4140/epoch_24.pth 8
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00N_fx4140.py 8
#GPU23 NOW

bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00N_fx1260.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00N/epoch_24.pth 8
bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00L_fx1100.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00L/epoch_24.pth 8
bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00L_fx880.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00L/epoch_24.pth 8
bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00K360_fx550.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_extended/debug_kitti360/epoch_24.pth 8
bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00A_fx1780.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00A/epoch_24.pth 8