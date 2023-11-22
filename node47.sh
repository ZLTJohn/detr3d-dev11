# bash tools/dist_train.sh projects/configs/bevdet_baseline/bevdet_r50_0.66nw.py 8
# bash tools/dist_train.sh projects/configs/bevdet_baseline/bevdet_r50_0.33w.py 8
# bash tools/dist_train.sh projects/configs/bevdet_baseline/bevdet_r50_0.33n.py 8
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00A_fx2070.py 8
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00K_fx2070.py 8
# bash tools/dist_train.sh projects/configs/mono_ablation/3.00ANL.py 8
# bash tools/dist_train.sh projects/configs/mono_ablation/3.50ANLK.py 8
#node48
# # bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN_0.1lr.py 8
# bash tools/dist_debug.sh /home/zhenglt/mmdev11/detr3d-dev11/projects/configs/bevdet_baseline/bevdet_r50_W_mono_bs8_smallres_noaug_syncBN_0.5lr.py 8

bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00A_fx1000_d140.py 8
bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00L_fx1000_d140.py 8
bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00K_fx1000_d140.py 8
bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00K360_fx1000_d140.py 8