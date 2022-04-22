
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
    'pev': 'potential_evaporation', 
    'slhf': 'surface_latent_heat_flux',
    'ssr': 'surface_net_solar_radiation', 
    'sp': 'surface_pressure', 
    'sshf': 'surface_sensible_heat_flux',
    'e': 'total_evaporation', 
    'tp': 'total_precipitation'
}

class ERA5(provider_base.Provider):

    def __init__(self, bands = ['t2m', 'pev', 'slhf', 'ssr', 'sp', 'sshf', 'e', 'tp'], aggregation_types = ["mean", "min", "max"], zarrpath = None, zarrurl = None):
        
        self.bands = bands
        self.aggregation_types = aggregation_types
        self.zarrpath = zarrpath
        self.zarrurl = zarrurl

    def load_data(self, bbox, time_interval):
        
        # If an URL is given, loads the cloud zarr, otherwise loads from local zarrpath
        if self.zarrurl:
            xr.open_zarr(fsspec.get_mapper(self.zarrurl), consolidated=True)
        elif self.zarrpath:
            era5 = xr.open_zarr(self.zarrpath, consolidated = False)

        era5 = era5[self.bands]

        center_lon = (bbox[0] + bbox[2])/2
        center_lat = (bbox[1] + bbox[3])/2

        era5 = era5.sel(lat = center_lat, lon = center_lon, method = "nearest").drop_vars(["lat", "lon"])

        agg_era5_collector = []
        for aggregation_type in self.aggregation_types:
            if aggregation_type == "mean":
                curr_agg_era5 = era5.groupby("time.date").mean("time").rename({"date": "time"})
            elif aggregation_type == "min":
                curr_agg_era5 = era5.groupby("time.date").min("time").rename({"date": "time"})
            elif aggregation_type == "max":
                curr_agg_era5 = era5.groupby("time.date").max("time").rename({"date": "time"})
            elif aggregation_type == "median":
                curr_agg_era5 = era5.groupby("time.date").median("time").rename({"date": "time"})
            elif aggregation_type == "std":
                curr_agg_era5 = era5.groupby("time.date").std("time").rename({"date": "time"})
            else:
                continue
            curr_agg_era5["time"] = np.array([str(d) for d in curr_agg_era5.time.values], dtype="datetime64[D]")
            curr_agg_era5 = curr_agg_era5.rename({b: f"era5_{b}_{aggregation_type}" for b in self.bands})
            agg_era5_collector.append(curr_agg_era5)
        
        agg_era5 = xr.merge(agg_era5_collector)

        agg_era5 = agg_era5.sel(time = slice(time_interval[:10], time_interval[-10:]))

        for b in self.bands:
            for a in self.aggregation_types:

                agg_era5[f"era5_{b}_{a}"].attrs = {
                    "provider": "ERA5-Land",
                    "interpolation_type": "linear",
                    "description": f"{SHORT_TO_LONG_NAMES[b]} 3-hourly data aggregated by {a}"
                }
        

        return agg_era5