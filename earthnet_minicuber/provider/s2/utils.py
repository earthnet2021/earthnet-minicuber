"""
This code is originally Core code for the sen2flux dataset from https://github.com/davemlz/sen2flux. It is licensed under:

The MIT License (MIT)

Copyright (c) 2022 David Montero Loaiza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import xml.etree.ElementTree as ET

import numpy as np

import requests

import xarray as xr
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def towerFootprint(x, y, distance, latlng=True, resolution=10):
    """Creates a Bounding Box given a pair of coordinates and a buffer distance.

    Parameters
    ----------
    x : float
        Longitude.
    y: float
        Latitude.
    distance : float
        Buffer distance in meters.
    latlng : bool, default = True
        Whether to return the Bounding Box as geographic coordinates.
    resolution : int | float, default = 10
        Spatial resolution to use.

    Returns
    -------
    dict
        Bounding Box returned as a GeoJSON.
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(x, y, x, y),
    )

    utm_crs = CRS.from_epsg(utm_crs_list[0].code)

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:" + utm_crs_list[0].code, always_xy=True
    )
    inverse_transformer = Transformer.from_crs(
        "EPSG:" + utm_crs_list[0].code, "EPSG:4326", always_xy=True
    )
    newCoords = transformer.transform(x, y)

    newCoords = [round(i / resolution) * resolution for i in newCoords]

    E = newCoords[0] + distance
    W = newCoords[0] - distance
    N = newCoords[1] + distance
    S = newCoords[1] - distance

    polygon = [
        [W, S],
        [E, S],
        [E, N],
        [W, N],
        [W, S],
    ]

    if latlng:
        polygon = [list(inverse_transformer.transform(x[0], x[1])) for x in polygon]

    footprint = {
        "type": "Polygon",
        "coordinates": [polygon],
    }

    return footprint


def towerCoordinates(x, y):
    """Transforms a pair of coordinates to UTM.

    Parameters
    ----------
    x : float
        Longitude.
    y: float
        Latitude.

    Returns
    -------
    list
        XY UTM coordinates.
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(x, y, x, y),
    )

    utm_crs = CRS.from_epsg(utm_crs_list[0].code)

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:" + utm_crs_list[0].code, always_xy=True
    )
    inverse_transformer = Transformer.from_crs(
        "EPSG:" + utm_crs_list[0].code, "EPSG:4326", always_xy=True
    )
    newCoords = transformer.transform(x, y)

    return newCoords


class Metadata:
    def __init__(self, signedMetadata):

        self.metadata = signedMetadata
        self.ET = ET.fromstring(requests.get(signedMetadata).content)

        GeometricInfo = self.ET.findall(
            "{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd}Geometric_Info"
        )[0]

        TileGeocoding = GeometricInfo.findall("Tile_Geocoding")[0]
        TileAngles = GeometricInfo.findall("Tile_Angles")[0]

        Geoposition = TileGeocoding.findall("Geoposition")[0]

        self.ULX = int(Geoposition.findall("ULX")[0].text)
        self.ULY = int(Geoposition.findall("ULY")[0].text)

        self.epsg = int(TileGeocoding.findall("HORIZONTAL_CS_CODE")[0].text[5:])

        bands_dict = {
            "Sun": {"Zenith": [], "Azimuth": []},
            "0": {"Zenith": [], "Azimuth": []},
            "1": {"Zenith": [], "Azimuth": []},
            "2": {"Zenith": [], "Azimuth": []},
            "3": {"Zenith": [], "Azimuth": []},
            "4": {"Zenith": [], "Azimuth": []},
            "5": {"Zenith": [], "Azimuth": []},
            "6": {"Zenith": [], "Azimuth": []},
            "7": {"Zenith": [], "Azimuth": []},
            "8": {"Zenith": [], "Azimuth": []},
            "9": {"Zenith": [], "Azimuth": []},
            "10": {"Zenith": [], "Azimuth": []},
            "11": {"Zenith": [], "Azimuth": []},
            "12": {"Zenith": [], "Azimuth": []},
        }

        SunAnglesGrid = TileAngles.findall("Sun_Angles_Grid")[0]

        def getValues(xmlElement, angle):
            angle_values = []
            for values in (
                xmlElement.findall(angle)[0].findall("Values_List")[0].findall("VALUES")
            ):
                angle_values.append(values.text.split(" "))
            return np.array(angle_values).astype(float)

        bands_dict["Sun"]["Zenith"] = getValues(SunAnglesGrid, "Zenith")
        bands_dict["Sun"]["Azimuth"] = getValues(SunAnglesGrid, "Azimuth")

        Viewing_Incidence_Angles_Grids = TileAngles.findall(
            "Viewing_Incidence_Angles_Grids"
        )

        for grid in Viewing_Incidence_Angles_Grids:
            band = grid.attrib["bandId"]
            zenith_values = getValues(grid, "Zenith")
            azimuth_values = getValues(grid, "Azimuth")
            bands_dict[band]["Zenith"].append(zenith_values)
            bands_dict[band]["Azimuth"].append(azimuth_values)

        for band in [str(x) for x in range(0, 13)]:
            bands_dict[band]["Zenith"] = np.nanmean(
                np.array(bands_dict[band]["Zenith"]), axis=0
            )
            bands_dict[band]["Azimuth"] = np.nanmean(
                np.array(bands_dict[band]["Azimuth"]), axis=0
            )

        self.angles = bands_dict

        grid_x_5000, grid_y_5000 = np.mgrid[
            self.ULX : self.ULX + 5000 * 23 : 5000,
            self.ULY : self.ULY - 5000 * 23 : -5000,
        ]

        self.grid_x = grid_x_5000 + 2500
        self.grid_y = grid_y_5000 - 2500

        bands_array = []
        bands_names = []

        names_lookup = {
            "Sun": "Sun",
            "0": "B01",
            "1": "B02",
            "2": "B03",
            "3": "B04",
            "4": "B05",
            "5": "B06",
            "6": "B07",
            "7": "B08",
            "8": "B8A",
            "9": "B09",
            "10": "B10",
            "11": "B11",
            "12": "B12",
        }

        for key, item in bands_dict.items():
            bands_array.append(item["Zenith"])
            bands_array.append(item["Azimuth"])
            bands_names.append(names_lookup[key] + "_Zenith")
            bands_names.append(names_lookup[key] + "_Azimuth")

        da = xr.DataArray(
            bands_array,
            dims=["band", "y", "x"],
            coords={
                "band": bands_names,
                "x": self.grid_x[:, 0],
                "y": self.grid_y[0, :],
            },
        )

        da_x = da.interpolate_na(dim=('x'), method='linear', fill_value = "extrapolate", use_coordinate = False)
        da_y = da.interpolate_na(dim=('y'), method='linear', fill_value = "extrapolate", use_coordinate = False)


        self.xr = 0.5 * da_x + 0.5 * da_y
        #.to_dataset("band").interpolate_na(method="linear", fill_value="extrapolate").to_array("band")
