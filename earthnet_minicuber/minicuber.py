

import numpy as np
import pandas as pd
import xarray as xr

#import earthnet_minicuber
from .provider import PROVIDERS

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from pathlib import Path

import pystac_client
import rasterio
import time

from copy import deepcopy

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

#PROVIDERS = {}#provider.PROVIDERS


LICENSE = """
License
EarthNet2022 - Africa is shared under CC-BY-NC-SA 4.0. You can understand our license in plain english: https://tldrlegal.com/license/creative-commons-attribution-noncommercial-sharealike-4.0-international-(cc-by-nc-sa-4.0). 

CC BY-NC-SA 4.0 License

Copyright (c) 2021 Max-Planck-Institute for Biogeochemistry, Vitus Benson, Jeran Poehls, Christian-Requena-Mesa, Nuno Carvalhais, Markus Reichstein

This data is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0

This Dataset uses ERA5 reanalysis data (2016-2021).

The Copernicus programme is governed by Regulation (EU) No 377/2014 of the European Parliament and of the Council of 3 April 2014 establishing the Copernicus programme and repealing Regulation (EU) No 911/2010. Within the Copernicus programme, a portfolio of land monitoring activities has been delegated by the European Union to the EEA. The land monitoring products and services are made available through the Copernicus land portal on a principle of full, open and free access, as established by the Copernicus data and information policy Regulation (EU) No 1159/2013 of 12 July 2013.

The Copernicus data and information policy is in line with the EEA policy of open and easy access to the data, information and applications derived from the activities described in its management plan.

See the full license at https://apps.ecmwf.int/datasets/licences/era5/ .

This Dataset uses Copernicus Sentinel data (2016-2022).

The access and use of Copernicus Sentinel Data and Service Information is regulated under EU law.1 In particular, the law provides that users shall have a free, full and open access to Copernicus Sentinel Data2 and Service Information without any express or implied warranty, including as regards quality and suitability for any purpose. 3 EU law grants free access to Copernicus Sentinel Data and Service Information for the purpose of the following use in so far as it is lawful4 : (a) reproduction; (b) distribution; (c) communication to the public; (d) adaptation, modification and combination with other data and information; (e) any combination of points (a) to (d). EU law allows for specific limitations of access and use in the rare cases of security concerns, protection of third party rights or risk of service disruption. By using Sentinel Data or Service Information the user acknowledges that these conditions are applicable to him/her and that the user renounces to any claims for damages against the European Union and the providers of the said Data and Information. The scope of this waiver encompasses any dispute, including contracts and torts claims, that might be filed in court, in arbitration or in any other form of dispute settlement.

This Dataset uses Esri 2020 land cover (https://www.arcgis.com/home/item.html?id=d6642f8a4f6d4685a24ae2dc0c73d4ac).
Esri 2020 land cover was produced by Impact Observatory for Esri. © 2021 Esri. Esri 2020 land cover is available under a Creative Commons BY-4.0 license and any copy of or work based on Esri 2020 land cover requires the following attribution: Esri 2020 land cover is based on the dataset produced for the Dynamic World Project by National Geographic Society in partnership with Google and the World Resources Institute.

This Dataset uses ESA WorldCover (https://esa-worldcover.org/en/data-access) under CC BY 4.0. We acknowledge the ESA WorldCover data.
© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium.

This Dataset uses SMAP L4 Global data (https://nsidc.org/data/smap)
As a condition of using these data, you must cite the use of this data set using the following citation. For more information, see our https://nsidc.org/about/use_copyright.html.

Reichle, R., G. De Lannoy, R. D. Koster, W. T. Crow, J. S. Kimball, and Q. Liu. 2021. SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Geophysical Data, Version 6. [Indicate subset used]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. doi: https://doi.org/10.5067/08S1A6811J0U. [Date Accessed].

This Dataset uses Geomorpho90m data (https://portal.opentopography.org/dataspace/dataset?opentopoID=OTDS.012020.4326.1)

We acknowledge the Geomorpho90m data. Amatulli, G., McInerney, D., Sethi, T., Strobl, P., Domisch, S. (2020). Geomorpho90m - Global High-Resolution Geomorphometry Layers. Distributed by OpenTopography. https://doi.org/10.5069/G91R6NPX. Accessed: 2021-12-17

This Dataset uses SRTM DEM data under CC-BY-4.0 (https://docs.digitalearthafrica.org/en/latest/data_specs/SRTM_DEM_specs.html). We acknowledge the SRTM DEM data.
See: T.G., Rosen, P. A., Caro, E., Crippen, R., Duren, R., Hensley, S., Kobrick, M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S., Shimada, J., Umland, J., Werner, M., Oskin, M., Burbank, D., & Alsdorf, D. (2007). The Shuttle Radar Topography Mission. In Reviews of Geophysics (Vol. 45, Issue 2). American Geophysical Union (AGU). https://doi.org/10.1029/2005rg000183

This Dataset uses Digital Earth Africa Landsat NDVI climatology data under CC-BY-4.0 (https://docs.digitalearthafrica.org/en/latest/data_specs/NDVI_Climatology_specs.html). We acknowledge the Digital Earth Africa Landsat NDVI climatology data.
"""


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

        # self.primary_provider = PROVIDERS[specs["primary_provider"]["name"]](**specs["primary_provider"]["kwargs"])

        # self.other_providers = [PROVIDERS[p["name"]](**p["kwargs"]) for p in specs["other_providers"]]

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

        #if not verbose:
        import warnings
        warnings.filterwarnings('ignore')

        self = cls(specs)

        all_data = []
        cube = None
        for time_interval in self.monthly_intervals:

            # if verbose:
            #     print(f"Loading {self.specs['primary_provider']['name']} for {time_interval}")

            # product_cube = self.primary_provider.load_data(self.padded_bbox, time_interval)

            # if product_cube is not None:
            #     cube = self.regrid_product_cube(product_cube)

            for i, provider in enumerate(self.providers):

                if verbose:
                    print(f"Loading {self.specs['providers'][i]['name']} for {time_interval}")

                product_cube = provider.load_data(self.padded_bbox, time_interval, full_time_interval = self.time_interval)

                if product_cube is not None:
                    if cube is None:
                        cube = self.regrid_product_cube(product_cube)
                    else:
                        cube = xr.merge([cube, self.regrid_product_cube(product_cube)])
                else:
                    if verbose:
                        print(f"Skipping {self.specs['providers'][i]['name']} for {time_interval} - no data found.")
            
            if cube is not None:
                if compute:
                    if verbose:
                        print(f"Downloading for {time_interval}...")
                    all_data.append(cube.compute())
                else:
                    all_data.append(cube)
            cube = None
        
        cube = xr.merge(all_data)

        cube['time'] = pd.DatetimeIndex(cube['time'].values)

        cube = cube.sel(time = slice(self.time_interval[:10], self.time_interval[-10:]))

        cube.attrs = {
            "dataset_name": "EarthNet2022 - Africa",
            "dataset_name_short": "en22",
            "dataset_version": "v0.2",
            "description": "This is a minicube from the EarthNet2022 - Africa dataset, created withing the context of the DeepCube Project. For more see https://www.earthnet.tech/ and https://deepcube-h2020.eu/.",
            "license": LICENSE,
            "provided_by": "Max-Planck-Institute for Biogeochemistry"
        }

        return cube



    @staticmethod
    def save_minicube_netcdf(minicube, savepath):

        savepath = Path(savepath)

        #encoding = {v: {"zlib": True, "complevel": 9} for v in list(minicube.variables)}

        encoding = {}
        for v in list(minicube.variables):
            if v in ["time", "time_clim"]:
                continue
            elif v.startswith("pq") or v in ["QA_PIXEL", "SCL", "mask"]:
                scale_factor, add_offset = 1.0, 0.0
            else:
                scale_factor, add_offset = compute_scale_and_offset(minicube[v])

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
                print(f"Cant read file.. {err}... skipping {pars['savepath']}")
                done = True
            except RuntimeError as err:
                print(f"Runtime error.. {err}... skipping {pars['savepath']}")
                done = True
            except IndexError as err:
                print(f"Index error..  {err}... skipping {pars['savepath']}")
                done = True
            except TypeError as err:
                print(f"Type error.. {err}... skipping {pars['savepath']}")
                done = True
            except ValueError as err:
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
            "primary_provider": {
            "name": "s2",
            "kwargs": {"bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"],#, "B09", "B11", "B12"], 
            "best_orbit_filter": True, "brdf_correction": True, "cloud_mask": True, "aws_bucket": "element84"}
            },
            "other_providers": [
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
                    "kwargs": {"bands": ['t2m', 'pev', 'slhf', 'ssr', 'sp', 'sshf', 'e', 'tp'], "aggregation_types": ["mean", "min", "max"], "zarrpath": "/Net/Groups/BGI/scratch/DeepCube/UC1/era5_africa/era5_africa_0d1_3hourly.zarr"}
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
