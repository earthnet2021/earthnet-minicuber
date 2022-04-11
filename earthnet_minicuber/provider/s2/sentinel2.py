
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

        ee.Initialize()


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

            stack = stackstac.stack(items_s2, epsg = epsg, assets = self.bands, dtype = "float32", properties = ["sentinel:data_coverage", "sentinel:sequence","sentinel:product_id"], band_coords = False, bounds_latlon = bbox, xy_coords = 'center')#.to_dataset("band")
            
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

            return stack

            # add epsg
            # add geotransform

            # add stackitemsearch

            # add stack raw load <-- here reproject/resample already?? <-- mmh possibly not bc need to match with cloud mask... In general is unclear how to exactly make sure all providers use the exact same grid.. decision to be made: do i want the primaryprovider grid or do i want a regularly-spaced lat-lon grid?... probably makes more sense to get the latter.
            # Question then is, do I actually reproject the data? Or do i just get the data in original epsg, then transform the coords to become lat-lon and then interpolate them onto a regular lat-lon grid of the correct shape? The latter is actually what Reprojection would also do, but only if i use 2d interpolation, or?? then maybe i should aim for 2d interpolation if possible...
            # So.. basically load with stackstack and lonlat bounds, use center coordinates. Dont use any regridding yet.. 


            # add best orbit filter:
            # - load metadata
            # - get data polygon.. intersect that with actual bbox. Find area of intersection. 
            # (- load 5 days for 1 band and check how many pixels are missing)
            # - filter by best orbit

            # add brdf correction:
            # - load metadata
            # - for bands that exist and can be corrected: correct

            # median for days with >1 obs

            # add davids cloud mask:
            # - generate geojson for earthengine
            # - download data from ee in small chunks
            
