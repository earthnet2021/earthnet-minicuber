

import numpy as np
import pandas as pd
import xarray as xr

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from pathlib import Path

import pystac_client
import rasterio
import time
import warnings
import traceback


from copy import deepcopy

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from .provider import PROVIDERS
from .license import LICENSE

def compute_scale_and_offset(da, n=16):
    """Calculate offset and scale factor for int conversion

    Based on Krios101's code above.
    """

    vmin = np.min(da).item()
    vmax = np.max(da).item()

    # stretch/compress data to the available packed range
    scale_factor = (vmax - vmin) / (2 ** n - 1)

    # translate the range to be symmetric about zero
    add_offset = vmin + 2 ** (n - 1) * scale_factor

    return scale_factor, add_offset

class Minicuber:

    def __init__(self, specs):
        self.specs = specs

        self.lon_lat = specs["lon_lat"]
        self.xy_shape = specs["xy_shape"]
        self.resolution = specs["resolution"]
        self.time_interval = specs["time_interval"]

        if "primary_provider" in specs:
            specs["providers"] =  [specs["primary_provider"]] + specs["other_providers"]

        self.providers = [PROVIDERS[p["name"]](**p["kwargs"]) for p in specs["providers"]]

        self.temporal_providers = [p for p in self.providers if p.is_temporal]
        self.spatial_providers = [p for p in self.providers if not p.is_temporal]


    @property
    def monthly_intervals(self):
        start = pd.Timestamp(self.time_interval[:10])
        end = pd.Timestamp(self.time_interval[-10:])
        monthly_intervals = []
        monthstart = start
        monthend = monthstart + pd.offsets.MonthEnd()
        while (monthend < end-pd.Timedelta("15 days")):
            if (monthend - monthstart) < pd.Timedelta("15 days"):
                monthend = monthend + pd.Timedelta("1 days") + pd.offsets.MonthEnd()
            if monthend > end-pd.Timedelta("15 days"):
                break
            monthly_intervals.append(monthstart.strftime('%Y-%m-%d') + "/" + monthend.strftime('%Y-%m-%d'))
            monthstart = monthend + pd.Timedelta("1 days")
            monthend = monthstart + pd.offsets.MonthEnd()
        monthly_intervals.append(monthstart.strftime('%Y-%m-%d') + "/" + end.strftime('%Y-%m-%d'))
        return monthly_intervals

    @property
    def bbox(self):

        utm_epsg = int(query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(self.lon_lat[0], self.lon_lat[1], self.lon_lat[0], self.lon_lat[1])
        )[0].code)

        transformer = Transformer.from_crs(4326, utm_epsg, always_xy=True)

        x_center, y_center = transformer.transform(*self.lon_lat)

        nx, ny = self.xy_shape
        
        x_left, x_right = x_center - self.resolution * (nx//2), x_center + self.resolution * (nx//2)

        y_top, y_bottom = y_center + self.resolution * (ny//2), y_center - self.resolution * (ny//2)

        return transformer.transform_bounds(x_left, y_bottom, x_right, y_top, direction = 'INVERSE') # left, bottom, right, top

    @property
    def padded_bbox(self):
        left, bottom, right, top = self.bbox
        lat_extra = (top - bottom) / self.xy_shape[0] * 6
        lon_extra = (right - left) / self.xy_shape[1] * 6
        return left - lon_extra, bottom - lat_extra, right + lon_extra, top + lat_extra


    @property
    def lon_lat_grid(self):
        nx, ny = self.xy_shape
        lon_left, lat_bottom, lon_right, lat_top = self.bbox

        lon_grid = np.linspace(lon_left, lon_right, nx)
        lat_grid = np.linspace(lat_top, lat_bottom, ny)

        return lon_grid, lat_grid

    def regrid_product_cube(self, product_cube):

        if ("x" in product_cube.coords) and ("y" in product_cube.coords):

            x, y = product_cube.x.values, product_cube.y.values

            product_epsg = product_cube.attrs["epsg"]

            transformer = Transformer.from_crs(4326, product_epsg, always_xy=True)

            lon_grid, lat_grid = self.lon_lat_grid

            new_x, new_y = transformer.transform(lon_grid, lat_grid)

            
            product_cube_nearest = product_cube.filter_by_attrs(interpolation_type=lambda v: ((v is None) or (v == "nearest")))
            product_cube_linear = product_cube.filter_by_attrs(interpolation_type="linear")
            if len(product_cube_nearest) > 0:
                product_cube_nearest = product_cube_nearest.interp(x = new_x, y = new_y, method = "nearest")
            if len(product_cube_linear) > 0:
                product_cube_linear = product_cube_linear.interp(x = new_x, y = new_y, method = "linear")
            if (len(product_cube_nearest) > 0) and (len(product_cube_linear) > 0):
                product_cube = xr.merge([product_cube_nearest, product_cube_linear])
            elif (len(product_cube_linear) > 0):
                product_cube = product_cube_linear
            else:
                product_cube = product_cube_nearest
            

            product_cube["x"], product_cube["y"] = lon_grid, lat_grid

            product_cube = product_cube.rename({"x": "lon", "y": "lat"})

        elif ("lat" in product_cube.coords) and ("lon" in product_cube.coords):
            lon_grid, lat_grid = self.lon_lat_grid
            product_cube_nearest = product_cube.filter_by_attrs(interpolation_type=lambda v: ((v is None) or (v == "nearest")))
            if len(product_cube_nearest) > 0:
                product_cube_nearest = product_cube_nearest.interp(lon = lon_grid, lat = lat_grid, method = "nearest")
            product_cube_linear = product_cube.filter_by_attrs(interpolation_type="linear")
            if len(product_cube_linear) > 0:
                product_cube_linear = product_cube_linear.interp(lon = lon_grid, lat = lat_grid, method = "linear")
            if (len(product_cube_nearest) > 0) and (len(product_cube_linear) > 0):
                product_cube = xr.merge([product_cube_nearest, product_cube_linear])
            elif (len(product_cube_linear) > 0):
                product_cube = product_cube_linear
            else:
                product_cube = product_cube_nearest
        
        product_cube.attrs = {}

        return product_cube



    @classmethod
    def load_minicube(cls, specs, verbose = True, compute = False):

        self = cls(specs)

        if not compute and (len(self.monthly_intervals) > 3):
            warnings.warn("You are querying a long time interval with compute = False, this might lead to failure in the dask sheduler and high memory consumption upon calling .compute(). Consider using compute = True instead.")

        warnings.filterwarnings('ignore')

        all_data = []
        cube = None
        for time_interval in self.monthly_intervals:

            for provider in self.temporal_providers:

                if verbose:
                    print(f"Loading {provider.__class__.__name__} for {time_interval}")

                product_cube = provider.load_data(self.padded_bbox, time_interval, full_time_interval = self.time_interval)

                if product_cube is not None:
                    if cube is None:
                        cube = self.regrid_product_cube(product_cube)
                    else:
                        cube = xr.merge([cube, self.regrid_product_cube(product_cube)])
                else:
                    if verbose:
                        print(f"Skipping {provider.__class__.__name__} for {time_interval} - no data found.")
            
            if cube is not None:
                if compute:
                    if verbose:
                        print(f"Downloading for {time_interval}...")
                    all_data.append(cube.compute())
                else:
                    all_data.append(cube)
            cube = None
        
        cube = xr.merge(all_data)

        for provider in self.spatial_providers:
            if verbose:
                print(f"Loading {provider.__class__.__name__}")
            product_cube = provider.load_data(self.padded_bbox, "not_needed")
            if product_cube is not None:
                if cube is None:
                    cube = self.regrid_product_cube(product_cube)
                else:
                    cube = xr.merge([cube, self.regrid_product_cube(product_cube)])
            else:
                if verbose:
                    print(f"Skipping {provider.__class__.__name__} - no data found.")
        
            

        cube['time'] = pd.DatetimeIndex(cube['time'].values)

        cube = cube.sel(time = slice(self.time_interval[:10], self.time_interval[-10:]))

        cube.attrs = {
            "dataset_name": "EarthNet2022 - Africa",
            "dataset_name_short": "en22",
            "dataset_version": "v0.3",
            "description": "This is a minicube from the EarthNet2022 - Africa dataset, created within the context of the DeepCube Project. For more see https://www.earthnet.tech/ and https://deepcube-h2020.eu/.",
            "license": LICENSE,
            "provided_by": "Max-Planck-Institute for Biogeochemistry"
        }

        return cube



    @staticmethod
    def save_minicube_netcdf(minicube, savepath):

        savepath = Path(savepath)

        encoding = {}
        for v in list(minicube.variables):
            if v in ["time", "time_clim"]:
                continue
            elif ("interpolation_type" in minicube[v].attrs) and (minicube[v].attrs["interpolation_type"] == "linear"):
                scale_factor, add_offset = compute_scale_and_offset(minicube[v])
            else:
                scale_factor, add_offset = 1.0, 0.0

            if abs(scale_factor) < 1e-8:
                encoding[v] = {"zlib": True, "complevel": 9}
            else:
                encoding[v] =  {
                    "dtype": 'int16',
                    "scale_factor": scale_factor,
                    "add_offset": add_offset,
                    "_FillValue": -32767,
                    "zlib": True,
                    "complevel": 9
                }

        if savepath.is_file():
            savepath.unlink()
        else:
            savepath.parents[0].mkdir(exist_ok=True, parents=True)

        minicube.to_netcdf(savepath, encoding = encoding, compute = True)

    @classmethod
    def save_minicube(cls, specs, savepath, verbose = True):

        minicube = cls.load_minicube(specs, verbose = verbose, compute = True)            

        if verbose:
            print(f"Downloading minicube at {specs['lon_lat']}")

        minicube = minicube.compute()

        if verbose:
            print(f"Saving minicube at {specs['lon_lat']}")

        cls.save_minicube_netcdf(minicube, savepath)


    @classmethod
    def save_minicube_mp(cls, pars):
        starttime = time.time()
        done = False
        c = 0
        while not done:
            try:
                cls.save_minicube(**pars)
                done = True
            except pystac_client.exceptions.APIError:
                time.sleep(10)
                cls.save_minicube(**pars)
            except rasterio._err.CPLE_OpenFailedError as err:
                traceback.print_exc()
                print(f"Cant read file.. {err}... skipping {pars['savepath']}")
                done = True
            except RuntimeError as err:
                traceback.print_exc()
                print(f"Runtime error.. {err}... skipping {pars['savepath']}")
                done = True
            except IndexError as err:
                traceback.print_exc()
                print(f"Index error..  {err}... skipping {pars['savepath']}")
                done = True
            except TypeError as err:
                traceback.print_exc()
                print(f"Type error.. {err}... skipping {pars['savepath']}")
                done = True
            except ValueError as err:
                traceback.print_exc()
                print(f"Value error.. {err}... skipping {pars['savepath']}")
                done = True
            c+=1
            if c > 5:
                done = True
                print(f"Pystac connection error.. skipping {pars['savepath']}")
        print(f"{pars['savepath']} took {time.time()-starttime:.2f} seconds.")

    
    @classmethod
    def create_minicubes_from_dataframe(cls, savepath, csvpath, n = 10):

        basespecs = {
            "lon_lat": (43.598946, 3.087414), # center pixel
            "xy_shape": (128,128), # width, height of cutout around center pixel
            "resolution": 30, # in meters.. will use this together with grid of primary provider..
            "time_interval": "2018-01-01/2021-12-31",
            "providers": [
                {
                    "name": "s2",
                    "kwargs": {"bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"],#, "B09", "B11", "B12"], 
                    "best_orbit_filter": True, "brdf_correction": True, "cloud_mask": True, "aws_bucket": "element84"}
                },
                {
                    "name": "s1",
                    "kwargs": {"bands": ["vv", "vh","mask"], "speckle_filter": True, "speckle_filter_kwargs": {"type": "lee", "size": 9}} 
                },
                {
                    "name": "ndviclim",
                    "kwargs": {"bands": ["mean", "std"]}
                },
                {
                    "name": "srtm",
                    "kwargs": {"bands": ["dem"]}
                },
                {
                    "name": "esawc",
                    "kwargs": {"bands": ["lc"]}
                },
                {
                    "name": "era5",
                    "kwargs": {
                        "bands": ['t2m', 'pev', 'slhf', 'ssr', 'sp', 'sshf', 'e', 'tp'], 
                        "aggregation_types": ["mean", "min", "max"], 
                        "zarrpath": "/Net/Groups/BGI/scratch/DeepCube/UC1/era5_africa/era5_africa_0d1_3hourly.zarr",
                        "zarrurl": "https://storage.de.cloud.ovh.net/v1/AUTH_84d6da8e37fe4bb5aea18902da8c1170/uc1-africa/era5_africa_0d1_3hourly.zarr",
                    }
                },
                {
                    "name": "sg",
                    "kwargs": {
                        "vars": ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "ocd", "sand", "silt", "soc"],
                        "depths": {"top": ["0-5cm", "5-15cm", "15-30cm"], "sub": ["30-60cm", "60-100cm", "100-200cm"]}, 
                        "vals": ["mean"],
                        "dirpath": "/Net/Groups/BGI/work_2/Landscapes_dynamics/downloads/soilgrids/Africa/"
                    }
                },
                {
                    "name": "geom",
                    "kwargs": {"filepath": "/Net/Groups/BGI/work_2/Landscapes_dynamics/downloads/Geomorphons/geom/geom_90M_africa_europe.tif"}
                },
                {
                    "name": "alos",
                    "kwargs": {}
                },
                {
                    "name": "cop",
                    "kwargs": {}
                }
                ]
        }

        savepath = Path(savepath)

        df = pd.read_csv(csvpath)

        all_pars = []
        for i in range(10,10+n):

            lon_lat_center = ((df.iloc[i]["MinLon"] + df.iloc[i]["MaxLon"])/2, (df.iloc[i]["MinLat"] + df.iloc[i]["MaxLat"])/2)

            specs = deepcopy(basespecs)
            specs["lon_lat"] = lon_lat_center

            name = df.iloc[i]["Unnamed: 0"]

            savepath_cube = savepath/f"{name}.nc"

            pars = {
                "specs": specs,
                "savepath": savepath_cube,
                "verbose": True
            }
            
            all_pars.append(pars)


        # for pars in tqdm(all_pars):
        #     cls.save_minicube_mp(pars)
    
        with ProcessPoolExecutor(max_workers=10) as pool:

            _ = list(tqdm(pool.map(cls.save_minicube_mp, all_pars),total = len(all_pars)))
