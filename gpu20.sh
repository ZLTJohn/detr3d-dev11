# bash tools/dist_test.sh projects/configs/mono_ablation/1.00K360_fx3100.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00K360_fx3100/epoch_24.pth 8
# bash tools/dist_test.sh projects/configs/mono_ablation/1.00L_fx4140.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00L_fx4140/epoch_24.pth 8
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00N_fx4140.py 8
#GPU23 NOW

# bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00N_fx1260.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00N/epoch_24.pth 8
# bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00L_fx1100.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00L/epoch_24.pth 8
# bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00L_fx880.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00L/epoch_24.pth 8
# bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00K360_fx550.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_extended/debug_kitti360/epoch_24.pth 8
# bash tools/dist_test.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00A_fx1780.py /home/zhenglt/mmdev11/detr3d-dev11/work_dirs_mono_ablate/1.00A/epoch_24.pth 8

# node51# bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_x+1.54_z+2.12.py 8
# node51# bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_x+1.54.py 8&&bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_x+3.py 8
# node48# bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_z-2.py 8
# bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_z+1.py 8
# bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_z+2.py 8
# bash tools/dist_train.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/mono_ablation/1.00W_fx2070_Ego_z+3.py 8
#gpu25
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_littleimgaug.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_nobevaug.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_noimgaug.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres.py 8
#node50
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN_recheck1.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN_recheck2.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN_1e-2decay_recheck1.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN_1e-2decay_recheck2.py 8

# bash tools/dist_train.sh projects/configs/mono_ablation/6.00ANWLKK360_fx2070_groundZ_optimalX_CPAsampler_front.py 8 #node46
# bash tools/dist_train.sh projects/configs/mono_ablation/5.00ANLKK360_fx2070_groundZ_optimalX_CPAsampler_front.py 8 #gpu22->gpu37->gpu22
# bash tools/dist_train.sh projects/configs/mono_ablation/3.50ANLK_fx2070_groundZ_optimalX_CPAsampler_front.py 8  # node47!
# bash tools/dist_train.sh projects/configs/mono_ablation/3.00ANL_fx2070_groundZ_optimalX_CPAsampler_front.py 8   # node48
# bash tools/dist_train.sh projects/configs/mono_ablation/2.00AN_fx2070_groundZ_optimalX_CPAsampler_front.py 8    #gpu37->node45?!
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00N_fx2070_groundZ_optimalX_CPAsampler_front.py 8 #gpu36
# # k360-direct ablation is on node51

# bash tools/dist_train.sh projects/configs/mono_ablation/1.00L_fx2070_CPAsampler_front.py 8 # gpu37
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00K360_fx2070_CPAsampler_front.py 8 # node45
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00A_fx1780_CPAsampler_front.py 8 # gpu23

# bash tools/dist_train.sh projects/configs/mono_ablation/1.00A_fx1780_groundZ_optimalX_CPAsampler_front.py 8 # node49
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00K_fx707_groundZ_optimalX_CPAsampler_front.py 8 # node51
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00K360_fx2070_groundZ_optimalX_CPAsampler_front.py 8 # node47
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00L_fx2070_groundZ_optimalX_CPAsampler_front.py 8 # gpu37


# bash tools/dist_train.sh projects/configs/bevdet_ablation/1.00W_recheck.py 8 # node45
# bash tools/dist_train.sh projects/configs/bevdet_ablation/1.00L_recheck.py 8 # node48
# bash tools/dist_train.sh projects/configs/bevdet_ablation/1.00K360_recheck.py 8 # gpu22
# bash tools/dist_train.sh projects/configs/bevdet_ablation/1.00K_recheck.py 8
# bash tools/dist_train.sh projects/configs/bevdet_ablation/1.00A_recheck.py 8 # node47

# bash tools/dist_train.sh projects/configs/bevdet_goodmono_withaug/1.00N_fx1000.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodmono_withaug/1.00A_fx1000.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodmono_withaug/1.00L_fx1000.py 8

# bash tools/dist_train.sh projects/configs/bevdet_goodrotate_withaug/2.00AN_fx1000_d90.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodrotate_withaug/1.00K_fx1000_d90.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodrotate_withaug/1.00K360_fx600_d90.py 8

bash tools/dist_train.sh projects/configs/bevdet_submission/1.00L_fx1000_d140.py 8 #node45
bash tools/dist_train.sh projects/configs/bevdet_submission/2.00NA_fx1000_d140.py 8