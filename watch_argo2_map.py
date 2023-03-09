from argparse import Namespace
from pathlib import Path

# path to where the logs live
dataroot = "data/argo2/train/"

# unique log identifier
log_id = "00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"
args = Namespace(**{"dataroot": Path(dataroot), "log_id": Path(log_id)})

import argparse
from pathlib import Path
from typing import List

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap, LaneSegment

# scaled to [0,1] for matplotlib.
PURPLE_RGB = [201, 71, 245]
PURPLE_RGB_MPL = np.array(PURPLE_RGB) / 255

DARK_GRAY_RGB = [40, 39, 38]
DARK_GRAY_RGB_MPL = np.array(DARK_GRAY_RGB) / 255
    
def single_log_teaser(args: argparse.Namespace) -> None:
    """
    For a single log, render all local crosswalks in green, and pedestrian crossings in purple,
    in a bird's eye view.
    """
    log_map_dirpath = Path(args.dataroot) / args.log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    x=1
    print(x)
    for _, ls in avm.vector_lane_segments.items():
        for i in ls.polygon_boundary:
            pt = np.array([1618.07879546,  263.04199128,   14.08253665])
            if ((pt-i)**2).sum() <10: 
                breakpoint()
        # right_ln_bnd
        # left_ln_bnd
        vector_plotting_utils.draw_polygon_mpl(ax, ls.polygon_boundary, color="g", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(ls.polygon_boundary, ax, color="g", alpha=0.2)

    # plot all pedestrian crossings
    for _, pc in avm.vector_pedestrian_crossings.items():
        vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="m", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(pc.polygon, ax, color="m", alpha=0.2)

    plt.savefig('debug/argo_map_vis')

single_log_teaser(args)