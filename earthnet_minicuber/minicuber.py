

import numpy as np

#import earthnet_minicuber
from .provider import PROVIDERS

from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

#PROVIDERS = {}#provider.PROVIDERS

class Minicuber:

    def __init__(self, specs):

        self.specs = specs

        self.lon_lat = specs["lon_lat"]
        self.xy_shape = specs["xy_shape"]
        self.resolution = specs["resolution"]
        self.time_interval = specs["time_interval"]

        self.primary_provider = PROVIDERS[specs["primary_provider"]["name"]](**specs["primary_provider"]["kwargs"])

        self.other_providers = [PROVIDERS[p["name"]](**p["kwargs"]) for p in specs["other_providers"]]

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

            product_cube = product_cube.interp(x = new_x, y = new_y, method = "nearest")

            product_cube["x"], product_cube["y"] = lon_grid, lat_grid

            product_cube = product_cube.rename({"x": "lon", "y": "lat"})

        elif ("lat" in product_cube.coords) and ("lon" in product_cube.coords):
            lon_grid, lat_grid = self.lon_lat_grid

            product_cube = product_cube.interp(lon = lon_grid, lat = lat_grid, method = "nearest") # mmh here get linear interpolation where applicable...
        
        product_cube.attrs = {}

        return product_cube



    @classmethod
    def load_minicube(cls, specs):

        self = cls(specs)

        product_cube = self.primary_provider.load_data(self.bbox, self.time_interval)

        cube = self.regrid_product_cube(product_cube)

        for provider in self.other_providers:

            product_cube = provider.load_data(self.bbox, self.time_interval)

            cube = xr.merge([cube, self.regrid_product_cube(product_cube)])

        return cube

