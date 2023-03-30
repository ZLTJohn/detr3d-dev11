import pandas
import copy
import mmengine
pkls = ['data/kitti-360/kitti360_infos_train.pkl','data/kitti-360/kitti360_infos_val.pkl']
for pkl in pkls:
    info = pandas.read_pickle(pkl)
    # info = pandas.read_pickle()
    for i in info['data_list']:
        CAM0 = copy.deepcopy(i['images']['CAM0'])
        i['images']['CAM1'] = CAM0
    mmengine.dump(info,pkl.replace('.pkl','_CAM0CAM0.pkl'))