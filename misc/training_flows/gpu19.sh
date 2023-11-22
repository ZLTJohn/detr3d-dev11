# bash tools/dist_train.sh projects/configs/mono_ablation/1.00W_fx3100.py 8
# bash tools/dist_train.sh projects/configs/mono_ablation/1.00K360_fx2070_gpu19.py 8

# bash tools/dist_test.sh projects/configs/mono_egotest/1.00W_CPAsampler_front.py work_dirs_mono_ablate/1.00W_fx2070_CPAsampler_front/epoch_24.pth 8
# bash tools/dist_test.sh projects/configs/mono_egotest/1.00W_groundZ_optimalX_CPAsampler_front.py work_dirs_mono_ablate/1.00W_fx2070_CPAsampler_front/epoch_24.pth 8
# bash tools/dist_test.sh projects/configs/mono_egotest/1.00W_groundZ_optimalX.py work_dirs_mono_ablate/1.00W_fx2070_recheck/epoch_24.pth 8

# bash tools/dist_train.sh projects/configs/bevdet_goodmono_withaug/1.00K_fx1000.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodmono_withaug/1.00W_fx1000.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodmono_withaug/1.00K360_fx1000.py 8

# bash tools/dist_train.sh projects/configs/bevdet_goodrotate_withaug/1.00N_fx1000_d90.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodrotate_withaug/1.00L_fx1000_d90.py 8
# bash tools/dist_train.sh projects/configs/bevdet_goodrotate_withaug/1.00A_fx1000_d90.py 8

# bash tools/dist_train.sh projects/configs/bevdet_submission/1.00A_fx1000_d140.py 8 #node49
# bash tools/dist_train.sh projects/configs/bevdet_submission/1.00K_fx1000_d140.py 8
# bash tools/dist_train.sh projects/configs/bevdet_submission/1.00K360_fx1000_d140.py 8

# bash tools/dist_train.sh projects/configs/bevdet_submission/3.00NAL_fx1000_d140.py 8 # gpu39

# bash tools/dist_train.sh projects/configs/bevdet_submission/3.50NALK_fx1000_d140.py 8 # gpu21
# bash tools/dist_train.sh projects/configs/bevdet_submission/5.00NALKK360_fx1000_d140.py 8 #gpu20
# bash tools/dist_train.sh projects/configs/bevdet_submission/6.00NALKK360W_fx1000_d140.py 8 @gpu22

#node47
# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00A_fx1000_d140.py 8
# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00L_fx1000_d140.py 8
# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00K_fx1000_d140.py 8
# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/1.00K360_fx1000_d140.py 8

bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/2.00NA_fx1000_d140.py 8 # gpu19
bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/3.00NAL_fx1000_d140.py 8

# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/3.50NALK_fx1000_d140.py 8 # node46

# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/5.00NALKK360_fx1000_d140.py 8 # gpu20

# bash tools/dist_train.sh projects/configs/bevdet_submission_egoalign/6.00NALKK360W_fx1000_d140.py 8 # gpu22