from typing import List, Sequence, Union
import random
import mmengine
from mmengine.registry import DATASETS
from mmengine.dataset.base_dataset import BaseDataset
from mmengine.dataset.dataset_wrapper import ConcatDataset

@DATASETS.register_module()
class CustomConcatDataset(ConcatDataset):
    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None,
                 dataset_ratios: List[float] = None,    # ratios of each dataset, e.g. 0.1 means decrease the original dataset to 10%
                 random_seed: int = 4):
        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')

        self.seed = random_seed
        if dataset_ratios is not None:
            assert len(dataset_ratios) == len(self.datasets)
            for i in range(len(dataset_ratios)):
                ratio = dataset_ratios[i]
                num_frame = len(self.datasets[i])
                idx = [id for id in range(num_frame)]
                random.Random(self.seed).shuffle(idx)
                idx = idx[:round(ratio*num_frame)]
                idx = sorted(idx)
                self.datasets[i] = self.datasets[i].get_subset(idx)
                print('new subset of {} has decreased frames from {} to {}'.format(
                        type(self.datasets[i]),num_frame, len(self.datasets[i])))

        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    raise ValueError(
                        f'{key} does not in the meta information of '
                        f'the {i}-th dataset')

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

from mmdet3d.datasets import NuScenesDataset
@DATASETS.register_module()
class CustomNusc(NuScenesDataset):
    # METAINFO = {
    #     'classes':
    #     ('Car', 'Pedestrian', 'Cyclist'),
    #     'version':
    #     'v1.0-trainval'
    # }
    # 'singapore-onenorth'
    # 'boston-seaport'
    # 'singapore-queenstown'
    # 'singapore-hollandvillage'
    def __init__(self,location=None,frame2city=None, **kwargs):
        self.load_interval = kwargs.pop('load_interval',1)
        self.location = location
        if frame2city is not None:
            self.frame2city = mmengine.load(frame2city)
        super().__init__(**kwargs)
    def filter_location(self, data_list):
        print('NuScenesDataset: location contains {} only!'.format(self.location))
        new_list = []
        for i in data_list:
            city_loc = self.frame2city[i['token']]
            if self.location in city_loc:
                new_list.append(i)
        return new_list

    def load_data_list(self) -> List[dict]:
        """Add the load interval."""
        data_list = super().load_data_list()
        data_list = data_list[::self.load_interval]
        if self.location is not None:
            data_list = self.filter_location(data_list)
        return data_list