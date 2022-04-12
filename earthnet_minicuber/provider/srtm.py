
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np


from . import provider_base


class SRTM(provider_base.Provider):

    def __init__(self, bands = ["dem"]):#, "mrrtf", "mrvbf", "slope"]):
        
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
            items_srtm = self.catalog.search(
                    bbox = bbox,
                    collections=["dem_srtm"]
                ).get_all_items()

            metadata = items_srtm.to_dict()['features'][0]["properties"]
            epsg = metadata["proj:epsg"]

            stack = stackstac.stack(items_srtm, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 512)
            stack["band"] = ["dem"]

            # if "mrrtf" in self.bands or "mrvbf" in self.bands or "slope" in self.bands:
            #     items_srtm = self.catalog.search(
            #             bbox = bbox,
            #             collections=["dem_srtm_deriv"]
            #         ).get_all_items()

            #     metadata = items_srtm.to_dict()['features'][0]["properties"]
            #     epsg = metadata["proj:epsg"]
                
            #     if stack is None:
            #         stack = stackstac.stack(items_srtm, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 512)
            #     else:
            #         deriv_stack = stackstac.stack(items_srtm, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 512)
            #         stack = xr.concat([stack, deriv_stack], dim = "band")

            # stack = stack.sel(band = self.bands)

            stack = stack.median("time")

            stack["band"] = [f"srtm_{b}" for b in stack.band.values]

            stack = stack.to_dataset("band")

            stack.attrs["epsg"] = epsg

            if "dem" in self.bands:
                stack["srtm_dem"].attrs = {"provider": "SRTM", "interpolation_type": "linear", "description": "Elevation data.", "units": "metre"}
            # if "mrvbf" in self.bands:
            #     stack["srtm_mrvbf"].attrs = {"provider": "SRTM", "interpolation_type": "linear", "description": "Multi-resolution Valley Bottom Flatness (MrVBF): this identifies valley bottoms (areas of deposition). Zero values indicate erosional terrain and values ≥1 and indicate progressively larger areas of deposition."}
            # if "mrrtf" in self.bands:
            #     stack["srtm_mrrtf"].attrs = {"provider": "SRTM", "interpolation_type": "linear", "description": "Multi-resolution Ridge Top Flatness (MrRTF): complementary to MrVBF, zero values indicate areas that are steep or low, and values ≥1 indicate progressively larger areas of high flat land."}
            # if "slope" in self.bands:
            #     stack["srtm_slope"].attrs = {"provider": "SRTM", "interpolation_type": "linear", "description": "Slope (percent): this is the rate of elevation change."}

            stack = stack.drop_vars(["epsg"])

            stack = stack.rename({"x": "lon", "y": "lat"})

            return stack




