

import os
import xarray as xr
import rioxarray as rioxr
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pathlib import Path

from . import provider_base



class Soilgrids(provider_base.Provider):

    SOILGRID_VARS = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "ocd", "sand", "silt", "soc"]

    SOILGRID_DEPTH = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

    DEPTH_DEPTHS = {
        "0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30, "60-100cm": 40, "100-200cm": 100
    }

    SOILGRID_VALS = ["mean", "uncertainty", "Q0.05", "Q0.5", "Q0.95"]

    SOILGRID_DESCS = {
        'bdod': 'Bulk density of the fine earth fraction',
        'cec': 'Cation Exchange Capacity of the soil',
        'cfvo': 'Volumetric fraction of coarse fragments (> 2 mm)',
        'clay': 'Proportion of clay particles (< 0.002 mm) in the fine earth fraction',
        'nitrogen': 'Total nitrogen (N)',
        'phh2o': 'Soil pH',
        'sand': 'Proportion of sand particles (> 0.05 mm) in the fine earth fraction',
        'silt': 'Proportion of silt particles (≥ 0.002 mm and ≤ 0.05 mm) in the fine earth fraction',
        'soc': 'Soil organic carbon content in the fine earth fraction',
        'ocd': 'Organic carbon density',
        'ocs': 'Organic carbon stocks'
        }

    def __init__(self, vars = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "ocd", "sand", "silt", "soc"], depths = {"0-30cm": ["0-5cm", "5-15cm", "15-30cm"], "30-200cm": ["30-60cm", "60-100cm", "100-200cm"]}, vals = ["mean", "uncertainty", "Q0.05", "Q0.5", "Q0.95"], dirpath = None):

        self.is_temporal = False
        
        self.vars = vars
        if isinstance(depths, list):
            depths = {d: [d] for d in depths}
        self.depths = depths
        self.vals = vals

        self.dirpath = Path(dirpath) if dirpath is not None else None


    def construct_url(self, var, depth, val):
        layer = f"{var}_{depth}_{val}"
        sg_layer = f'{layer.split("_")[0]}/{layer}.vrt'
        location = f'https://files.isric.org/soilgrids/latest/data/{sg_layer}'
        sg_url = f'/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url={location}'
        return sg_url

    def open_one_soilgrid(self, var, depth, val, bbox):

        if self.dirpath is not None:
            filepath = self.dirpath/f"sg_africa_{var}_{depth}_{val}.tif"
            if filepath.is_file():
                da = rioxr.open_rasterio(filepath)
                da = da.rename({"x": "lon", "y": "lat"})
                da = da.sel(x = slice(bbox[0], bbox[2]), y = slice(bbox[3], bbox[1])).isel(band = 0).drop_vars(["band"], errors = "ignore")
                da = da.astype("float32")

                da = da.where(lambda x: x != -32768.0)

                return da.rename(f"sg_{var}_{depth}_{val}")
        
        sg_url = self.construct_url(var, depth, val)

        with rasterio.open(sg_url) as src:
            with WarpedVRT(src, crs=4326, resampling=Resampling.nearest) as vrt:
                dst_window = vrt.window(*bbox)
                data = vrt.read(window=dst_window)

        _, ny, nx = data.shape
        lon_left, lat_bottom, lon_right, lat_top = bbox

        lon_grid = np.linspace(lon_left, lon_right, nx)
        lat_grid = np.linspace(lat_top, lat_bottom, ny)

        da = xr.DataArray(data[0,...], coords = {"lat": lat_grid, "lon": lon_grid}, dims = ("lat", "lon"))

        # da = rioxr.open_rasterio(WarpedVRT(rasterio.open(sg_url), crs = 4326, resampling = Resampling.nearest))#, chunks = (1, 50, 50))

        # da = da.sel(x = slice(bbox[0], bbox[2]), y = slice(bbox[3], bbox[1]))#.compute()

        da = da.astype("float32")

        da = da.where(lambda x: x != -32768.0)

        # return da.isel(band = 0).rename(f"sg_{var}_{depth}_{val}").drop_vars(["band"], errors = "ignore")

        return da.rename(f"sg_{var}_{depth}_{val}")



    def load_data(self, bbox, time_interval, **kwargs):
        
        arrays = []
        for var in self.vars:

            for val in self.vals:

                for depth_agg, depths in self.depths.items():

                    curr_arrays = None

                    total = 0

                    for depth in depths:

                        factor = self.DEPTH_DEPTHS[depth]

                        total += factor

                        curr_arr = factor * self.open_one_soilgrid(var, depth, val, bbox)

                        if curr_arrays is None:
                            curr_arrays = curr_arr
                        else:
                            curr_arrays += curr_arr

                    curr_arrays /= total

                    arrays.append(curr_arrays.rename(f"sg_{var}_{depth_agg}_{val}"))
        
        stack = xr.merge(arrays)

        stack = stack.drop_vars(["spatial_ref"], errors = "ignore")

        for var in stack.data_vars:
            sgvar, depth, val = var.split("_")[1:]
            stack[var].attrs = {"provider": "Soilgrids", "interpolation_type": "nearest", "description": self.SOILGRID_DESCS[sgvar] + f". Prediction value: {val}. Depth: {depth}."}
                    
        stack.attrs["epsg"] = 4326

        return stack




