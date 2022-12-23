_base_ = ['./detr3d_res101_gridmask_dev-1.x.py']
model = dict(type='Detr3D_old')

# before fixing h,w bug in feature-sampling
# mAP: 0.3450
# mATE: 0.7740
# mASE: 0.2675
# mAOE: 0.3960
# mAVE: 0.8737
# mAAE: 0.2156
# NDS: 0.4198
# Eval time: 161.5s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.534   0.565   0.152   0.071   0.907   0.214
# truck   0.285   0.839   0.213   0.114   0.984   0.229
# bus     0.346   0.924   0.199   0.117   2.060   0.379
# trailer 0.166   1.108   0.230   0.551   0.734   0.126
# construction_vehicle    0.082   1.057   0.446   1.013   0.125   0.387
# pedestrian      0.426   0.688   0.294   0.508   0.459   0.195
# motorcycle      0.343   0.696   0.260   0.475   1.268   0.180
# bicycle 0.275   0.691   0.275   0.578   0.452   0.015
# traffic_cone    0.521   0.555   0.314   nan     nan     nan
# barrier 0.473   0.619   0.293   0.138   nan     nan

# after fixing h,w bug in feature-sampling
# mAP: 0.3469
# mATE: 0.7651
# mASE: 0.2678
# mAOE: 0.3916
# mAVE: 0.8758
# mAAE: 0.2110
# NDS: 0.4223
# Eval time: 117.2s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.546   0.544   0.152   0.070   0.911   0.208
# truck   0.286   0.834   0.212   0.113   1.005   0.231
# bus     0.346   0.870   0.196   0.116   2.063   0.383
# trailer 0.167   1.106   0.233   0.549   0.687   0.093
# construction_vehicle    0.082   1.060   0.449   0.960   0.120   0.384
# pedestrian      0.424   0.700   0.295   0.512   0.462   0.194
# motorcycle      0.340   0.709   0.259   0.489   1.288   0.176
# bicycle 0.278   0.698   0.275   0.586   0.473   0.018
# traffic_cone    0.529   0.526   0.313   nan     nan     nan
# barrier 0.471   0.603   0.292   0.131   nan     nan
