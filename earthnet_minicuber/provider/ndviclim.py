
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np


from . import provider_base


class NDVIClim(provider_base.Provider):

    def __init__(self, bands = ["mean", "std", "count"]):
        self.is_temporal = False
        
        self.bands = bands

        URL = "https://explorer.digitalearth.africa/stac/"
        self.catalog = pystac_client.Client.open(URL)

        os.environ['AWS_NO_SIGN_REQUEST'] = "TRUE"
        os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'


    def load_data(self, bbox, time_interval, **kwargs):
        
        with rasterio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com'):
            items_clim = self.catalog.search(
                bbox = bbox,
                collections=["ndvi_climatology_ls"],
            ).get_all_items()

            metadata = items_clim.to_dict()['features'][0]["properties"]
            epsg = metadata["proj:epsg"]

            stack = stackstac.stack(items_clim, epsg = epsg, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 800)

            clims = {}
            if "mean" in self.bands:
                mean_clim = stack.sel(band = ['mean_jan', 'mean_feb', 'mean_mar', 'mean_apr', 'mean_may', 'mean_jun', 'mean_jul','mean_aug', 'mean_sep','mean_oct', 'mean_nov','mean_dec']).isel(time = 0).drop_vars(["time"]).rename({"band":"time_clim"})
                mean_clim["time_clim"] = np.array([np.datetime64(f"1970-{str(v).zfill(2)}-15") for v in range(1,13)])
                mean_clim.attrs = {"provider": "Landsat NDVI climatology", "interpolation_type": "linear", "description": "Mean Landsat NDVI (1984-2020)"}
                clims["ndviclim_mean"] = mean_clim.astype("float32")

            if "std" in self.bands:
                std_clim = stack.sel(band = ['stddev_jan', 'stddev_feb', 'stddev_mar', 'stddev_apr', 'stddev_may', 'stddev_jun', 'stddev_jul','stddev_aug', 'stddev_sep','stddev_oct', 'stddev_nov','stddev_dec']).isel(time = 0).drop_vars(["time"]).rename({"band":"time_clim"})
                std_clim["time_clim"] = np.array([np.datetime64(f"1970-{str(v).zfill(2)}-15") for v in range(1,13)])
                std_clim.attrs = {"provider": "Landsat NDVI climatology", "interpolation_type": "linear", "description": "Standard Deviation Landsat NDVI (1984-2020)"}
                clims["ndviclim_std"] = std_clim.astype("float32")
            
            if "count" in self.bands:
                count_clim = stack.sel(band = ['count_jan', 'count_feb', 'count_mar', 'count_apr', 'count_may', 'count_jun', 'count_jul','count_aug', 'count_sep','count_oct', 'count_nov','count_dec']).isel(time = 0).drop_vars(["time"]).rename({"band":"time_clim"})
                count_clim["time_clim"] = np.array([np.datetime64(f"1970-{str(v).zfill(2)}-15") for v in range(1,13)])
                count_clim.attrs = {"provider": "Landsat NDVI climatology", "interpolation_type": "nearest", "description": "Measurement count Landsat NDVI (1984-2020)"}
                clims["ndviclim_count"] = count_clim.astype("float32")

            clim = xr.Dataset(clims)

            clim = clim.drop_vars(["epsg", "id"])

            clim.attrs["epsg"] = epsg

            return clim



