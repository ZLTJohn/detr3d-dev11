node50
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/egoXY.py 8
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/egoAll+ManyCam.py 8
# bash tools/dist_train.sh projects/configs/ablation_study_3.00/Rt.py 8

node46:
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/egoXYZ.py 8
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/Rt+egoAll.py 8
bash tools/dist_train.sh projects/configs/ablation_study_3.00/All_but_singlecam.py 8

node48:
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/egoAll.py 8  
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/K.py 8
# bash tools/dist_train.sh projects/configs/ablation_study_3.00/KRt+egoAll+shiftaug.py 8
bash tools/dist_train.sh projects/configs/ablation_study_3.00/Rt.py 8 1day 15h

gpu38:
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/egoAll+shiftaug.py 8
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/KRt+shiftaug.py 8
# bash tools/dist_train.sh projects/configs/ablation_study_3.00/K+egoAll.py 8

gpu37:
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/KRt+egoAll.py 8
bash tools/dist_train.sh projects/configs/ablation_study_3.00/All.py 8

gpu39:
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/KRt.py 8
bash tools/dist_train.sh projects/configs/ablation_study_3.00/KRt+egoAll+ManyCam.py 8

node47:
done bash tools/dist_train.sh projects/configs/ablation_study_3.00/K+egoAll.py 8
bash tools/dist_train.sh projects/configs/ablation_study_3.00/KRt+egoAll+shiftaug.py 8 eta:15h from 0315-15:08