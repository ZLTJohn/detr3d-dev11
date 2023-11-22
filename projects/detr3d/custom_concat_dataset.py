from typing import List, Sequence, Union
import random
import mmengine
from mmengine.registry import DATASETS
from mmengine.dataset.base_dataset import BaseDataset
from mmengine.dataset.dataset_wrapper import ConcatDataset
from mmdet3d.datasets import NuScenesDataset, WaymoDataset, LyftDataset, KittiDataset

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
    def __init__(self,locations=None, **kwargs):
        self.load_interval = kwargs.pop('load_interval',1)
        self.locations = locations
        super().__init__(**kwargs)

    def filter_location(self, data_list):
        print('NuScenesDataset: location contains {} only!'.format(self.locations))
        new_list = []
        for i in data_list:
            if i['city_name'] in self.locations:
                new_list.append(i)
        return new_list
    
    def add_dataset_name(self,data_list):
        for i in data_list:
            i['dataset_name'] = self.metainfo['dataset']
        return data_list

    def load_data_list(self) -> List[dict]:
        """Add the load interval."""
        data_list = super().load_data_list()
        data_list = data_list[::self.load_interval]
        data_list = self.add_dataset_name(data_list)
        if self.locations is not None:
            data_list = self.filter_location(data_list)
        return data_list
    
@DATASETS.register_module()
class CustomWaymo(WaymoDataset):
# location_other
# location_phx
# location_sf
    def __init__(self,locations=None, **kwargs):
        self.locations = locations
        super().__init__(**kwargs)
    
    def filter_location(self, data_list):
        print('WaymoDataset: location contains {} only!'.format(self.locations))
        new_list = []
        for i in data_list:
            if i['city_name'] in self.locations:
                new_list.append(i)
        return new_list
    
    def add_dataset_name(self,data_list):
        for i in data_list:
            i['dataset_name'] = self.metainfo['dataset']
        return data_list

    def load_data_list(self) -> List[dict]:
        """Add the load interval."""
        data_list = super(WaymoDataset,self).load_data_list()
        data_list = data_list[::self.load_interval]
        data_list = self.add_dataset_name(data_list)
        if self.locations is not None:
            data_list = self.filter_location(data_list)
        return data_list

@DATASETS.register_module()
class CustomLyft(LyftDataset):
    # Palo_Alto
    def __init__(self,locations=None, focal_interval = None, **kwargs):
        self.load_interval = kwargs.pop('load_interval',1)
        self.locations = locations
        self.focal_interval = focal_interval
        super().__init__(**kwargs)

    def filter_location(self, data_list):
        print('LyftDataset: location contains {} only!'.format(self.locations))
        new_list = []
        for i in data_list:
            if i['city_name'] in self.locations:
                new_list.append(i)
        return new_list
    
    def filter_focal_length(self, data_list):
        print('LyftDataset: focal_lengths contains {} only!'.format(self.focal_interval))
        new_list = []
        for i in data_list:
            focal = i['images']['CAM_FRONT']['cam2img'][0][0]
            if (self.focal_interval[0] <= focal) and (focal <= self.focal_interval[1]):
                new_list.append(i)
        return new_list
    
    def add_dataset_name(self,data_list):
        for i in data_list:
            i['dataset_name'] = self.metainfo['dataset']
        return data_list

    def load_data_list(self) -> List[dict]:
        """Add the load interval."""
        data_list = super().load_data_list()
        data_list = data_list[::self.load_interval]
        data_list = self.add_dataset_name(data_list)
        if self.locations is not None:
            data_list = self.filter_location(data_list)
        if self.focal_interval is not None:
            data_list = self.filter_focal_length(data_list)
        return data_list
    
@DATASETS.register_module()
class CustomKitti(KittiDataset):
    # TODO: add img_path for CAM3 in info['images']
    def __init__(self,locations=None, **kwargs):
        self.load_interval = kwargs.pop('load_interval',1)
        self.locations = locations
        super().__init__(**kwargs)

    # def filter_location(self, data_list):
    #     print('KittiDataset: location contains {} only!'.format(self.locations))
    #     new_list = []
    #     for i in data_list:
    #         if i['city_name'] in self.locations:
    #             new_list.append(i)
    #     return new_list
    
    def add_dataset_name(self,data_list):
        for i in data_list:
            i['dataset_name'] = self.metainfo['dataset']
            i['city_name'] = 'Germany'
        return data_list

    def load_data_list(self) -> List[dict]:
        """Add the load interval."""
        data_list = super().load_data_list()
        data_list = data_list[::self.load_interval]
        data_list = self.add_dataset_name(data_list)
        # if self.locations is not None:
        #     data_list = self.filter_location(data_list)
        return data_list