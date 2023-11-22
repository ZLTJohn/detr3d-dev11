import pandas
from glob import glob
kit = glob('data/kitti/*.pkl')
info_k = pandas.read_pickle(kit[0])
info_l = pandas.read_pickle('data/lyft/debug_val.pkl')
import mmengine
for pkl in kit:
    info = pandas.read_pickle(pkl)
    if 'trainval' in pkl:
        s = 'trainval'
    elif 'train' in pkl:
        s = 'train'
    elif 'val' in pkl:
        s = 'val'
    elif 'test' in pkl:
        s = 'test'
    info['metainfo']['version'] = 'v0.00-'+s
    mmengine.dump(info,pkl)