
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np
import planetary_computer as pc


from . import provider_base


class ALOSWorld(provider_base.Provider):

    def __init__(self):#, "mrrtf", "mrvbf", "slope"]):
        
        URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"
        self.catalog = pystac_client.Client.open(URL)


    def load_data(self, bbox, time_interval, **kwargs):
        
        stack = None

        search = self.catalog.search(
                bbox = bbox,
                collections=["alos-dem"]
            )
            
        items_dem = pc.sign(search)

        metadata = items_dem.to_dict()['features'][0]["properties"]
        epsg = metadata["proj:epsg"]

        stack = stackstac.stack(items_dem, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 512)

        stack["band"] = ["alos_dem"]

        stack = stack.median("time")

        stack = stack.to_dataset("band")

        stack["alos_dem"].attrs = {"provider": "ALOS World 3D-30m", "interpolation_type": "linear", "description": "Elevation data.", "units": "metre"}

        stack = stack.drop_vars(["epsg"])

        stack = stack.rename({"x": "lon", "y": "lat"})
        
        stack.attrs["epsg"] = epsg

        return stack




