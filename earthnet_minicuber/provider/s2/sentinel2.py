
import os
import pystac_client
import stackstac
import rasterio

import planetary_computer as pc
from rasterio import RasterioIOError
import time
import numpy as np
import xarray as xr
import random
from contextlib import nullcontext

from shapely.geometry import Polygon, box

from .sen2flux import sunAndViewAngles, computeNBAR
from .cloudmask import CloudMask, cloud_mask_reduce
from .. import provider_base

S2BANDS_DESCRIPTION = {
    "B01": "Coastal aerosol",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Red edge 1",
    "B06": "Red edge 2",
    "B07": "Red edge 3",
    "B08": "Near infrared (NIR) Broad",
    "B8A": "Near infrared (NIR) Narrow",
    "B09": "Water vapour",
    "B11": "Short-wave infrared (SWIR) 1",
    "B12": "Short-wave infrared (SWIR) 2",
    "AOT": "Aerosol optical thickness",
    "WVP": "Scene average water vapour",
    "SCL": "Scene classification layer",
    "mask": "Deep Learning Cloud Mask, trained by Vitus Benson on cloudSEN12, leveraging code from César Aybar."
}

class Sentinel2(provider_base.Provider):

    def __init__(self, bands = ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "WVP"], best_orbit_filter = True, five_daily_filter = False, brdf_correction = True, cloud_mask = True, aws_bucket = "dea", s2_avail_var = True):
        
        self.is_temporal = True

        self.cloud_mask = CloudMask(bands=bands) if cloud_mask else None

        if self.cloud_mask and "SCL" not in bands:
            bands += ["SCL"]

        self.bands = bands
        self.best_orbit_filter = best_orbit_filter
        self.five_daily_filter = five_daily_filter
        self.brdf_correction = brdf_correction
        self.aws_bucket = aws_bucket
        self.s2_avail_var = s2_avail_var

        if aws_bucket == "dea":
            URL = "https://explorer.digitalearth.africa/stac/"
            os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'

        elif aws_bucket == "planetary_computer":
            URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'

        else:
            URL = "https://earth-search.aws.element84.com/v0"
            if 'AWS_S3_ENDPOINT' in os.environ:
                del os.environ['AWS_S3_ENDPOINT']
        
        self.catalog = pystac_client.Client.open(URL)

        os.environ['AWS_NO_SIGN_REQUEST'] = "TRUE"

    def get_attrs_for_band(self, band):

        attrs = {}
        attrs["provider"] = "Sentinel 2"
        attrs["interpolation_type"] = "nearest" if band in ["SCL", "mask", "avail"] else "linear"
        attrs["description"] = S2BANDS_DESCRIPTION[band]
        if self.brdf_correction and band in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]:
            attrs["brdf_correction"] = "Nadir BRDF Adjusted Reflectance (NBAR)"
        if band == "SCL":
            attrs["classes"] = """
                            0 - No data
                            1 - Saturated / Defective
                            2 - Dark Area Pixels
                            3 - Cloud Shadows
                            4 - Vegetation
                            5 - Bare Soils
                            6 - Water
                            7 - Clouds low probability / Unclassified
                            8 - Clouds medium probability
                            9 - Clouds high probability
                            10 - Cirrus
                            11 - Snow / Ice
                            """
        elif band == "mask":
            attrs["classes"] = """
            0 - free sky
            1 - cloud
            2 - cloud shadows
            3 - snow
            4 - masked other reasons
            """

        return attrs
        


    def load_data(self, bbox, time_interval, **kwargs):

        if self.aws_bucket == "dea":
            cm = rasterio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com')
            
        else:
            cm = nullcontext()

        gdal_session = stackstac.DEFAULT_GDAL_ENV.updated(always=dict(session=rasterio.session.AWSSession(aws_unsigned = True, endpoint_url = 's3.af-south-1.amazonaws.com' if self.aws_bucket == "dea" else None)))

        with cm as gs:
        

            search = self.catalog.search(
                        bbox = bbox,
                        collections=["s2_l2a" if self.aws_bucket == "dea" else ("sentinel-2-l2a" if self.aws_bucket == "planetary_computer" else "sentinel-s2-l2a-cogs")],
                        datetime=time_interval
                    )
            
            if self.aws_bucket == "planetary_computer":
                for attempt in range(10):
                    try:
                        items_s2 = pc.sign(search)
                    except pystac_client.exceptions.APIError:
                        print(f"Sen2: Planetary computer time out, attempt {attempt}, retrying in 60 seconds...")
                        time.sleep(random.uniform(30,90))
                    else:
                        break
                else:
                    print("Loading Sen2 failed after 10 attempts...")
                    return None
            else:
                items_s2 = search.get_all_items()

            if len(items_s2.to_dict()['features']) == 0:
                return None

            metadata = items_s2.to_dict()['features'][0]["properties"]
            epsg = metadata["proj:epsg"]


            stack = stackstac.stack(items_s2, epsg = epsg, assets = self.bands, dtype = "float32", properties = ["sentinel:product_id"], band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 2048,errors_as_nodata=(RasterioIOError('.*'), ), gdal_env=gdal_session)


            if self.aws_bucket != "planetary_computer":
                stack = stack.rename({"id": "id_old"}).rename({"sentinel:product_id": "id"})

            stack = stack.drop_vars(["id_old", "sentinel:data_coverage", "sentinel:sequence"], errors = "ignore")

            stack.attrs["epsg"] = epsg

            if self.best_orbit_filter:
                
                if "full_time_interval" in kwargs:
                    full_time_interval = kwargs["full_time_interval"]

                    search_best_orbit = self.catalog.search(
                            bbox = bbox,
                            collections=["s2_l2a" if self.aws_bucket == "dea" else ("sentinel-2-l2a" if self.aws_bucket == "planetary_computer" else "sentinel-s2-l2a-cogs")],
                            datetime=full_time_interval
                        )

                    if self.aws_bucket == "planetary_computer":
                        items_s2_best_orbit = pc.sign(search_best_orbit)
                    else:
                        items_s2_best_orbit = search_best_orbit.get_all_items()
                else:
                    full_time_interval = time_interval
                    items_s2_best_orbit = items_s2

                bbox_poly = box(*bbox)
                area_and_dates = [(bbox_poly.intersection(Polygon(f['geometry']["coordinates"][0])).area, np.datetime64(f["properties"]["datetime"][:10])) for f in items_s2_best_orbit.to_dict()['features']]

                _, max_area_date = max(area_and_dates, key = lambda x: x[0])
                min_date, max_date = np.datetime64(full_time_interval[:10]), np.datetime64(full_time_interval[-10:])

                dates = np.arange(max_area_date - ((max_area_date - min_date)//5)*5, max_date+1, 5)

                stack = stack.sel(time = stack.time.dt.date.isin(dates))
            
            elif self.five_daily_filter:

                if "full_time_interval" in kwargs:
                    full_time_interval = kwargs["full_time_interval"]
                else:
                    full_time_interval = time_interval

                min_date, max_date = np.datetime64(full_time_interval[:10]), np.datetime64(full_time_interval[-10:])

                dates = np.arange(min_date, max_date+1, 5)

                stack = stack.sel(time = stack.time.dt.date.isin(dates))


            if self.brdf_correction:

                stack_plus_metadata, stack = sunAndViewAngles(search.get_all_items(), stack, aws_bucket = self.aws_bucket)

                if stack_plus_metadata is None:
                    print("Skipping, no valid metadata for BRDF correction")
                    return None
                
                stack = computeNBAR(stack_plus_metadata, stack)


            if self.cloud_mask:
                stack = self.cloud_mask(stack)
                    
            bands = stack.band.values
            stack["band"] = [f"s2_{b}" for b in stack.band.values]

            stack = stack.to_dataset("band")

            for band in bands:
                if band in ["AOT", "WVP"]:
                    stack[f"s2_{band}"] = (stack[f"s2_{band}"]/65535).astype("float32")
                elif band not in ["SCL","mask"]:
                    stack[f"s2_{band}"] = (stack[f"s2_{band}"]/10000).astype("float32")
            
            stack = stack.drop_vars(["epsg", "id", "id_old", "sentinel:data_coverage", "sentinel:sequence", "sentinel:product_id"], errors = "ignore")
            
            stack["time"] = np.array([str(d) for d in stack.time.values], dtype="datetime64[D]")

            if self.s2_avail_var:
                stack["s2_avail"] = xr.DataArray(np.ones_like(stack.time.values, dtype = "uint8"), coords = {"time": stack.time.values}, dims = ("time",))
            
            if len(stack.time) > 0:
                stack = stack.groupby("time.date").last(skipna = False).rename({"date": "time"})
            else:
                return None
            
            for band in bands:
                stack[f"s2_{band}"].attrs = self.get_attrs_for_band(band)
            
            stack.attrs["epsg"] = epsg

            return stack
