
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np
from contextlib import nullcontext
import planetary_computer as pc
import time
import random

from . import provider_base


class ESAWorldcover(provider_base.Provider):

    def __init__(self, bands = ["lc"],aws_bucket = "dea"):
        self.is_temporal = False
        self.bands = bands
        self.aws_bucket = aws_bucket

        if aws_bucket == "dea":
            URL = "https://explorer.digitalearth.africa/stac/"
            os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'

        else:#elif aws_bucket == "planetary_computer":
            URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'

        self.catalog = pystac_client.Client.open(URL)

        os.environ['AWS_NO_SIGN_REQUEST'] = "TRUE"


    def load_data(self, bbox, time_interval, **kwargs):

        if self.aws_bucket == "dea":
            cm = rasterio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com')
            
        else:
            cm = nullcontext()
        
        with cm:

            stack = None

            # if "dem" in self.bands:
            search = self.catalog.search(
                    bbox = bbox,
                    collections=["esa_worldcover" if self.aws_bucket == "dea" else "esa-worldcover"]
                )

            if self.aws_bucket == "planetary_computer":
                for attempt in range(10):
                    try:
                        items_esawc = pc.sign(search)
                    except pystac_client.exceptions.APIError:
                        print(f"ESAWC: Planetary computer time out, attempt {attempt}, retrying in 60 seconds...")
                        time.sleep(random.uniform(30,90))
                    else:
                        break
                else:
                    print("Loading ESAWC failed after 10 attempts...")
                    return None
            else:
                items_esawc = search.get_all_items()

            if len(items_esawc.to_dict()['features']) == 0:
                return None

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




