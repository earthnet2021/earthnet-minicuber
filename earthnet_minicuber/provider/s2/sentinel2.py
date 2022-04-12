
import os
import pystac_client
import stackstac
import rasterio
import ee

import numpy as np
import xarray as xr

from shapely.geometry import Polygon, box

#import earthnet_minicuber
#from ..provider import Provider
from .sen2flux import sunAndViewAngles, computeNBAR, computeCloudMask
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
    "mask": "sen2flux Cloud Mask"
}

class Sentinel2(provider_base.Provider):

    def __init__(self, bands = ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "WVP"], best_orbit_filter = True, brdf_correction = True, cloud_mask = True):

        self.bands = bands
        self.best_orbit_filter = best_orbit_filter
        self.brdf_correction = brdf_correction
        self.cloud_mask = cloud_mask

        URL = "https://explorer.digitalearth.africa/stac/"
        self.catalog = pystac_client.Client.open(URL)

        os.environ['AWS_NO_SIGN_REQUEST'] = "TRUE"
        os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'

        if self.cloud_mask:
            ee.Initialize()


    def get_attrs_for_band(self, band):

        attrs = {}
        attrs["provider"] = "Sentinel 2"
        attrs["interpolation_type"] = "nearest" if band in ["SCL", "mask"] else "linear"
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
            """

        return attrs
        


    def load_data(self, bbox, time_interval):
        
        with rasterio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com'):

            items_s2 = self.catalog.search(
                        bbox = bbox,
                        collections=["s2_l2a"],
                        datetime=time_interval
                    ).get_all_items()
            
            metadata = items_s2.to_dict()['features'][0]["properties"]
            epsg = metadata["proj:epsg"]
            # geotransform = metadata["proj:transform"]

            stack = stackstac.stack(items_s2, epsg = epsg, assets = self.bands, dtype = "float32", properties = ["sentinel:data_coverage", "sentinel:sequence","sentinel:product_id"], band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 256)#.to_dataset("band")
            
            stack = stack.rename({"id": "id_old"}).rename({"sentinel:product_id": "id"})
            
            stack.attrs["epsg"] = epsg

            if self.best_orbit_filter:

                s2_poly = Polygon(items_s2.to_dict()['features'][-1]['geometry']["coordinates"][0])
                bbox_poly = box(*bbox)
                area_and_dates = [(bbox_poly.intersection(Polygon(f['geometry']["coordinates"][0])).area, np.datetime64(f["properties"]["datetime"][:10])) for f in items_s2.to_dict()['features']]

                _, max_area_date = max(area_and_dates, key = lambda x: x[0])
                min_date, max_date = np.datetime64(time_interval[:10]), np.datetime64(time_interval[-10:])

                dates = np.arange(max_area_date - ((max_area_date - min_date)//5)*5, max_date+1, 5)

                stack = stack.sel(time = stack.time.dt.date.isin(dates))

            
            if self.brdf_correction:

                stack_plus_metadata, stack = sunAndViewAngles(items_s2, stack)
                
                stack = computeNBAR(stack_plus_metadata, stack)

            stack = stack.isel(time = [v[0] for v in stack.groupby("time.date").groups.values()])
            #stack = stack.groupby("time.date").last(skipna = False).rename({"date": "time"})

            if self.cloud_mask:

                E, S, W, N = bbox

                polygon = [
                    [W, S],
                    [E, S],
                    [E, N],
                    [W, N],
                    [W, S],
                ]

                aoi = {
                    "type": "Polygon",
                    "coordinates": [polygon],
                }
                
                if len(stack.time.values) < 10:
                    stack = computeCloudMask(aoi, stack, stack.time.values[0].astype('datetime64[Y]').astype(int) + 1970)
                else:
                    cm_chunks = []
                    for i in range(0, len(stack.time.values), 10):
                        cm_chunks.append(computeCloudMask(aoi, stack.isel(time = slice(i,i+10)), stack.isel(time = i).time.values.astype('datetime64[Y]').astype(int) + 1970))

                    stack = xr.concat(cm_chunks, dim = "time")

            # TODO
            # Clean Sentinel 2
            # Rescale 0-1 ?? 
            # Fix s.t. for non-categorical layers uses linear interpolation at reprojection
            # Optimize chunking

            bands = stack.band.values
            stack["band"] = [f"s2_{b}" for b in stack.band.values]

            stack = stack.to_dataset("band")

            for band in bands:
                if band in ["AOT", "WVP"]:
                    stack[f"s2_{band}"] = (stack[f"s2_{band}"]/65535).astype("float32")
                elif band not in ["SCL","mask"]:
                    stack[f"s2_{band}"] = (stack[f"s2_{band}"]/10000).astype("float32")
                stack[f"s2_{band}"].attrs = self.get_attrs_for_band(band)
            
            stack = stack.drop_vars(["epsg", "id", "id_old", "sentinel:data_coverage", "sentinel:sequence"])

            return stack
