
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np
import planetary_computer as pc
import time
import random

from . import provider_base


class Copernicus30(provider_base.Provider):

    def __init__(self):#, "mrrtf", "mrvbf", "slope"]):

        self.is_temporal = False
        
        URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
        self.catalog = pystac_client.Client.open(URL)


    def load_data(self, bbox, time_interval, **kwargs):
        
        stack = None

        search = self.catalog.search(
                bbox = bbox,
                collections=["cop-dem-glo-30"]
            )
            
        for attempt in range(10):
            try:
                items_dem = pc.sign(search)
            except pystac_client.exceptions.APIError:
                print(f"COP30 Dem: Planetary computer time out, attempt {attempt}, retrying in 60 seconds...")
                time.sleep(random.uniform(30,90))
            else:
                break
        else:
            print("Loading COP30 DEM failed after 10 attempts...")
            return None

        if len(items_dem.to_dict()['features']) == 0:
            return None

        metadata = items_dem.to_dict()['features'][0]["properties"]
        epsg = metadata["proj:epsg"]

        stack = stackstac.stack(items_dem, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 512)

        stack["band"] = ["cop_dem"]

        stack = stack.median("time")

        stack = stack.to_dataset("band")

        stack["cop_dem"].attrs = {"provider": "Copernicus DEM GLO-30", "interpolation_type": "linear", "description": "Elevation data.", "units": "metre"}

        stack = stack.drop_vars(["epsg"])

        stack = stack.rename({"x": "lon", "y": "lat"})
        
        stack.attrs["epsg"] = epsg

        return stack




