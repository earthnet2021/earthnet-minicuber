

import os
import xarray as xr
import rioxarray as rioxr
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pathlib import Path

from . import provider_base



class Geomorphons(provider_base.Provider):

    def __init__(self, filepath):
        self.is_temporal = False
        self.filepath = filepath

    def load_data(self, bbox, time_interval, **kwargs):
        
        stack = rioxr.open_rasterio(self.filepath)

        stack = stack.sel(x = slice(bbox[0], bbox[2]), y = slice(bbox[3], bbox[1])).assign_coords({"band": ["geom_cls"]}).to_dataset("band")

        stack = stack.astype("float32")

        stack = stack.where(lambda x: x != 0.)

        stack = stack.drop_vars(["spatial_ref"], errors = "ignore")
        
        stack["geom_cls"].attrs = {"provider": "Geomorpho90m", "interpolation_type": "nearest", "description": "Geomorphon classes. Original resolution ~90m. For more see: https://www.nature.com/articles/s41597-020-0479-6",
        "classes": """
        1: flat,
        2: summit,
        3: ridge,
        4: shoulder,
        5: spur,
        6: slope,
        7: hollow,
        8: footslope,
        9: valley,
        10: depression
        """}
                    
        stack.attrs["epsg"] = 4326

        return stack




