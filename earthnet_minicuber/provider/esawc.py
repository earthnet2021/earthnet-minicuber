
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np


from . import provider_base


class ESAWorldcover(provider_base.Provider):

    def __init__(self, bands = ["lc"]):#, "mrrtf", "mrvbf", "slope"]):
        
        # TODO: Fix srtm_deriv bands.. these are not in lat-lon but some weird other projection..
        self.bands = bands

        URL = "https://explorer.digitalearth.africa/stac/"
        self.catalog = pystac_client.Client.open(URL)

        os.environ['AWS_NO_SIGN_REQUEST'] = "TRUE"
        os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'


    def load_data(self, bbox, time_interval):
        
        with rasterio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com'):

            stack = None

            # if "dem" in self.bands:
            items_esawc = self.catalog.search(
                    bbox = bbox,
                    collections=["esa_worldcover"]
                ).get_all_items()

            metadata = items_esawc.to_dict()['features'][0]["properties"]
            epsg = metadata["proj:epsg"]

            stack = stackstac.stack(items_esawc, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 1024)
            stack["band"] = ["lc"]


            stack = stack.median("time")

            stack["band"] = [f"esawc_{b}" for b in stack.band.values]

            stack = stack.to_dataset("band")


            if "lc" in self.bands:
                stack["esawc_lc"].attrs = {"provider": "ESA Worldcover", "interpolation_type": "nearest", "description": "Land cover classification", "classes": """
                10 - Tree cover
                20 - Shrubland
                30 - Grassland
                40 - Cropland
                50 - Built-up
                60 - Bare / sparse vegetation
                70 - Snow and Ice
                80 - Permanent water bodies
                90 - Herbaceous wetland
                95 - Mangroves
                100 - Moss and lichen
                """}

            stack = stack.drop_vars(["epsg"])

            stack = stack.rename({"x": "lon", "y": "lat"})
            
            stack.attrs["epsg"] = epsg

            return stack




