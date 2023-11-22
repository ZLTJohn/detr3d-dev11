import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import torch
import os
from glob import glob
from mmengine import ProgressBar
import mmengine
context = None
# context = '10289507859301986274_4200_000_4220_000'
# timestamp = '1557847410072052'
# obj_id = 'MZ9L0O68d7aa3OveM57wFw'
val_path = 'data/waymo_dev1x/waymo_format/training/'
fnames = os.listdir(val_path)
tfnames= glob(val_path+'*.tfrecord')
frame2env_info={}
pb = ProgressBar(len(tfnames))
for name in tfnames:
    dataset = tf.data.TFRecordDataset(name, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        ts = frame.timestamp_micros
        env_info = {
            'time_of_day': frame.context.stats.time_of_day,
            'location': frame.context.stats.location,
            'weather': frame.context.stats.weather
        }
        frame2env_info[ts] = env_info
    pb.update()
mmengine.dump(frame2env_info, 'data/waymo_dev1x/timestamp2envinfo_train.pkl')