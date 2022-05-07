
import os
import pystac_client
import stackstac
import rasterio
import xarray as xr
import numpy as np
import fsspec

from . import provider_base

SHORT_TO_LONG_NAMES = {
    't2m': '2m_temperature', 
    'slhf': 'surface_latent_heat_flux',
    'ssr': 'surface_net_solar_radiation', 
    'msl': 'mean_sea_level_pressure', 
    'sshf': 'surface_sensible_heat_flux',
    'e': 'evaporation', 
    'tp': 'total_precipitation',
}

subdaily_vars = ['2m_temperature', 'mean_sea_level_pressure']

class ECMWF(provider_base.Provider):

    def __init__(self, 
                 bands = ['t2m', 'slhf', 'ssr', 'msl', 'sshf', 'e', 'tp'], 
                 daily_aggregation_types = ["mean", "min", "max"], 
                 leadtime_months = [0,1,2,3,4,5], 
                 ensemble_realization_numbers = [0,1,2,3,20,40,45,50], 
                 ensemble_aggregation = ["mean","median","min", "max"], 
                 zarrpath = None, 
                 zarrurl = None):
        
        self.bands = bands
        self.aggregation_types = aggregation_types
        self.zarrpath = zarrpath
        self.zarrurl = zarrurl

    def load_data(self, bbox, time_interval):
        
        # If an URL is given, loads the cloud zarr, otherwise loads from local zarrpath
        if self.zarrurl:
            xr.open_zarr(fsspec.get_mapper(self.zarrurl), consolidated=True)
        elif self.zarrpath:
            ecmwf = xr.open_zarr(self.zarrpath, consolidated = False)

        ecmwf = ecmwf[self.bands]

        center_lon = (bbox[0] + bbox[2])/2
        center_lat = (bbox[1] + bbox[3])/2

        ecmwf = ecmwf.sel(latitude = center_lat, longitude = center_lon, method = "nearest").drop_vars(["latitude", "longitude"])
    
        
        agg_ecmwf_collector = []
#         for leadtime_month in leadtime_months:
#             #WIP
            
        for daily_aggregation_type in self.daily_aggregation_types:
            if daily_aggregation_type == "mean":
                curr_agg_ecmwf = ecmwf.groupby("step.date").mean("step").rename({"date": "time"})
            elif daily_aggregation_type == "min":
                curr_agg_ecmwf = ecmwf.groupby("step.date").min("step").rename({"date": "time"})
            elif daily_aggregation_type == "max":
                curr_agg_ecmwf = ecmwf.groupby("step.date").max("step").rename({"date": "time"})
            elif daily_aggregation_type == "median":
                curr_agg_ecmwf = ecmwf.groupby("step.date").median("step").rename({"date": "time"})
            elif daily_aggregation_type == "std":
                curr_agg_ecmwf = ecmwf.groupby("step.date").std("step").rename({"date": "time"})
            else:
                continue
            curr_agg_ecmwf["step"] = np.array([str(d) for d in curr_agg_ecmwf.step.values], dtype="datetime64[D]")
            curr_agg_ecmwf = curr_agg_ecmwf.rename({b: f"ecmwf_{b}_{daily_aggregation_type}" for b in self.bands})
            agg_ecmwf_collector.append(curr_agg_ecmwf)

        agg_ecmwf = xr.merge(agg_ecmwf_collector)

        agg_ecmwf = agg_ecmwf.sel(time = slice(time_interval[:10], time_interval[-10:]))

        for b in self.bands:
            for a in self.aggregation_types:

                agg_ecmwf[f"ecmwf_{b}_{a}"].attrs = {
                    "provider": "ECMFWF seasonal weather forecasts",
                    "interpolation_type": "linear",
                    "description": f"{SHORT_TO_LONG_NAMES[b]} 3-hourly data aggregated by {a}"
                }
    

        return agg_ecmwf