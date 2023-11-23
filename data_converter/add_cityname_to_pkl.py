import pandas
import mmengine
from glob import glob
nusc_token2city = 'data/nus_v2/token2citylocation.pkl'
waymo_ts2env_info = 'data/waymo_dev1x/timestamp2envinfo_trainval.pkl'
nusc_mp = pandas.read_pickle(nusc_token2city)
waymo_mp = pandas.read_pickle(waymo_ts2env_info)

def get_cityname(ds, frame):
    if 'arg' in ds:
        city = frame['city_name']
    elif 'nus' in ds:
        city = nusc_mp[frame['token']]
    elif 'way' in ds:
        city = waymo_mp[frame['timestamp']]['location']

    return city
# ssd_pkls = glob('/localdata_ssd/nusc_dev1x/debug_val*') + glob('/localdata_ssd/nusc_dev1x/nuscenes_infos_*.pkl') + glob('/localdata_ssd/waymo_dev1x/*.pkl')
pkls = glob('data/nus_v2/debug_val*') + glob('data/waymo_dev1x/kitti_format/*.pkl') + glob('data/nus_v2/nuscenes_infos_*.pkl')
for pkl in pkls:
    if ('train' in pkl) or ('val' in pkl):
        info = pandas.read_pickle(pkl)
        for i in info['data_list']:
            city_name = get_cityname(pkl,i)
            i['city_name'] = city_name
        mmengine.dump(info, pkl)
        print('finished', pkl)
    else:
        print('skip',pkl)

lyft = glob('data/lyft/*.pkl')
for pkl in lyft:
    info = pandas.read_pickle(pkl)
    for i in info['data_list']:
        city_name = 'Palo_Alto'
        i['city_name'] = city_name
    mmengine.dump(info, pkl)
    print('finished', pkl)
