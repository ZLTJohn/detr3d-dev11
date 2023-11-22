import pandas
pkl = 'data/kitti-360/kitti360_infos_all.pkl'
info = pandas.read_pickle(pkl)
data_list = {}
info_train = []
cnt_train=0
info_val = []
cnt_val=0
for i in info['data_list']:
    id = i['log_id']
    if id not in data_list:
        data_list[id]  = []
    data_list[id].append(i)

# 80%-20%
for key in data_list:
    data = data_list[key]
    size = len(data)
    val = int(size * 0.8)
    for i in range(val):
        data[i]['sample_idx'] = cnt_train
        cnt_train += 1
        info_train.append(data[i])
    for i in range(val,len(data)):
        data[i]['sample_idx'] = cnt_val
        cnt_val += 1
        info_val.append(data[i])
import mmengine
info_train = {
    'metainfo': info['metainfo'],
    'data_list': info_train
}
info_val = {
    'metainfo': info['metainfo'],
    'data_list': info_val
}
mmengine.dump(info_train,'data/kitti-360/kitti360_infos_train.pkl')
mmengine.dump(info_val,'data/kitti-360/kitti360_infos_val.pkl')
