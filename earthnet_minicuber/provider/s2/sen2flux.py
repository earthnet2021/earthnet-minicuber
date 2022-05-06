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
from calendar import monthrange

import ee
import eemont
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio.features
import stackstac
import wxee
import xarray as xr
from .cloud_mask import *
from .nbar import nbar
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio import plot
from .utils import *
import requests
import time

# from dask.distributed import Client
# from dask_gateway import GatewayCluster


# def signAndStack(items, bbox, epsg):
#     """Signs the Planetary Computer items from the STAC and stacks them.

#     Parameters
#     ----------
#     items : dict
#         Items from the STAC.
#     bbox : dict
#         GeoJSON of the Bounding Box.
#     epsg : int
#         CRS.

#     Returns
#     -------
#     xarray.DataArray
#         Stack of signed items.
#     """
#     signed_items = []
#     for item in items:
#         item.clear_links()
#         signed_items.append(pc.sign(item).to_dict())

#     S2 = stackstac.stack(
#         signed_items,
#         assets=[
#             "B01",
#             "B02",
#             "B03",
#             "B04",
#             "B05",
#             "B06",
#             "B07",
#             "B08",
#             "B8A",
#             "B09",
#             "B11",
#             "B12",
#             "SCL",
#             "AOT",
#             "WVP",
#         ],
#         resolution=10,
#         bounds=bbox,
#         epsg=epsg,
#     ).where(
#         lambda x: x > 0, other=np.nan
#     )  # NO DATA IS ZERO -> THEN TRANSFORM ZEROS TO NO DATA

#     return S2


def sunAndViewAngles(items, ref, aws_bucket = "dea"):
    """Creates the Sun and Sensor View angles in degrees from the metadata.

    Parameters
    ----------
    items : dict
        Items from the STAC.
    ref : xarray.DataArray
        Stacked array to merge the angles.

    Returns
    -------
    xarray.DataArray, xarray.DataArray
        Stacked array with angles and the previous array.
    """
    items = items[::-1]
    metadata_items = []
    idx = []
    iC = 0
    for item in items:
        if ((aws_bucket != "planetary_computer") and (item.properties['sentinel:product_id'] in ref.id)) or (item.id in ref.id):
        # if item.properties['sentinel:product_id'] in ref.id:
            item.clear_links()
            try:
                if aws_bucket == "planetary_computer":
                    metadata_items.append(
                        Metadata(pc.sign(item.assets["granule-metadata"].href))
                    )
                else:
                    metadata_items.append(
                        Metadata(item.assets["metadata"].href)
                    )
                idx.append(iC)
            except:
                print("Skip one corrupted image!")
            iC = iC + 1
    ref = ref[idx]
    # metadata_items = []
    # for item in items:
    #    item.clear_links()
    #    metadata_items.append(Metadata(pc.sign(item.assets["granule-metadata"].href)))

    angles_array = []
    for metadata in metadata_items:

        curr_metadata = metadata.xr
        
        if ref.epsg.values != metadata.epsg:

            transformer = Transformer.from_crs(int(metadata.epsg), int(ref.epsg.values), always_xy = True)

            x_grid, y_grid = metadata.xr.x.values, metadata.xr.y.values

            new_x, new_y = transformer.transform(x_grid, y_grid)

            curr_metadata["x"] = new_x
            curr_metadata["y"] = new_y

        
        angles_array.append(
            curr_metadata.interp(
                x=ref.x.data,
                y=ref.y.data,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        ) #.rio.write_nodata(np.NaN, inplace=True).rio.write_crs(f"epsg:{ref.epsg.values}", inplace=True).rio.interpolate_na(method = "linear")

    md = xr.concat(angles_array, dim="time")
    md = md.assign_coords(time=("time", ref.time.data))

    # band_coords = {
    #     "title": ("band", md.band.data),
    #     "gsd": ("band", [10] * 28),
    #     "common_name": ("band", md.band.data),
    #     "center_wavelength": ("band", [None] * 28),
    #     "full_width_half_max": ("band", [None] * 28),
    # }

    # md = md.assign_coords(band_coords)

    complete = xr.concat([ref, md], dim="band")

    return complete, ref
    # if len(complete.time.data.tolist()) == len(set(complete.time.data.tolist())):
    #     COMPLETE_DAILY = complete
    # else:
    #     COMPLETE_DAILY = complete.groupby("time").median()

    # return COMPLETE_DAILY, ref


def computeNBAR(arr, ref):
    """Computes the NBAR.

    Parameters
    ----------
    arr : xarray.DataArray
        Stacked array with angles.
    ref : xarray.DataArray
        Previous stacked array.

    Returns
    -------
    xarray.DataArray
        Stacked array with NBAR.
    """
    complete_nbar = xr.concat([nbar(i) for i in arr], dim="time")
    complete_nbar = (complete_nbar * 10000).round()

    # band_coords = {
    #     "title": ("band", complete_nbar.band.data),
    #     "gsd": ("band", [10] * 9),
    #     "common_name": ("band", complete_nbar.band.data),
    #     "center_wavelength": ("band", [None] * 9),
    #     "full_width_half_max": ("band", [None] * 9),
    # }

    # complete_nbar = complete_nbar.assign_coords(band_coords)

    # if len(ref.time.data.tolist()) == len(set(ref.time.data.tolist())):
    #     GROUPED = ref
    # else:
    #     GROUPED = ref.groupby("time").median()

    nbar_bands = [b.split("_")[0] for b in complete_nbar.band.values]

    S2_NBAR_DAY = xr.concat([ref.where(~ref.band.isin(nbar_bands), drop = True), complete_nbar.assign_coords({"band": nbar_bands})], dim="band")

    return S2_NBAR_DAY

def cloud_mask_reduce(x, axis = None, **kwargs):
    return np.where((x==1).any(axis = axis), 1, np.where((x==2).any(axis = axis), 2, np.where((x==3).any(axis = axis), 3, np.where((x==0).any(axis = axis), 0, 4))))

def computeCloudMask(aoi, arr, year):
    """Computes the cloud mask in GEE and stacks it to the array.

    Parameters
    ----------
    aoi : dict
        GeoJSON of the Bounding Box.
    arr : xarray.DataArray
        Stacked array with NBAR.
    ref : xarray.DataArray
        Previous stacked array.
    year : int
        Year to look for in GEE.

    Returns
    -------
    xarray.DataArray
        Stacked array with the cloud mask.
    """
    iDate = f"{year}-01-01"
    eDate = f"{year + 1}-01-01"

    ee_aoi = ee.Geometry(aoi)

    GEE_filter = []
    if len(arr.time) == 1:
        if "band" in arr.id.dims:
            ids = [arr.id.isel(band=0).data[0]]
        else:
            ids = [str(arr.id.data)]
    else:
        if "band" in arr.id.dims:
            ids = arr.id.isel(band=0).data
        else:
            ids = arr.id.data
    for x in ids:
        x = x.split("_")
        if len(x) == 7:
            GEE_filter.append("_".join([x[2], x[5]]))
        else:
            GEE_filter.append("_".join([x[2], x[4]]))
    # try:
    #     for x in ids:
    #         x = x.split("_")
    #         if len(x) == 7:
    #             GEE_filter.append("_".join([x[2], x[5]]))
    #         else:
    #             GEE_filter.append("_".join([x[2], x[4]]))
    # except TypeError:
    #     breakpoint()
    
    def setFilter(img):
        x = img.id().split("_")
        return img.set({"PC_filter": ee.String(x.get(0)).cat("_").cat(x.get(2))})

    S2_ee = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterBounds(ee_aoi)
        .filterDate(iDate, eDate)
    )
    S2_ee = S2_ee.map(setFilter).filter(ee.Filter.inList("PC_filter", GEE_filter))

    CLOUD_MASK = (
        PCL_s2cloudless(S2_ee).map(PSL).map(PCSL).map(matchShadows).select("CLOUD_MASK")
    )

    downloaded = False
    c = 1
    while not downloaded:
        try:
            CLOUD_MASK_xarray = CLOUD_MASK.wx.to_xarray(
                scale=10, crs="EPSG:" + str(arr.attrs["epsg"]), region=ee_aoi, progress = False
            )
            downloaded = True
        except requests.exceptions.HTTPError:
            if c > 10:
                print("No Cloud mask for Sentinel IDs:")
                print(arr.id.data)
                return arr
            time.sleep(60 * c)
            c+=1
    
    CLOUD_MASK_xarray = CLOUD_MASK_xarray.where(lambda x: x >= 0, other=4)

    CLOUD_MASK = CLOUD_MASK_xarray.CLOUD_MASK.interp(
            x=arr.x.data,
            y=arr.y.data,
            method="nearest",
            kwargs={"fill_value": 4},#"extrapolate"},
        )
    # try:
    #     CLOUD_MASK = CLOUD_MASK_xarray.CLOUD_MASK.interp(
    #         x=arr.x.data,
    #         y=arr.y.data,
    #         method="nearest",
    #         kwargs={"fill_value": np.nan},#"extrapolate"},
    #     )
    # except AttributeError:
    #     breakpoint()
    #CLOUD_MASK = CLOUD_MASK.resample(time="1D").max().where(lambda x: x >= 0, drop=True)
    CLOUD_MASK = CLOUD_MASK.assign_coords(band="mask").expand_dims("band")

    # band_coords = {
    #     "title": "CLOUD_MASK",
    #     "gsd": 10,
    #     "common_name": "CLOUD_MASK",
    #     "center_wavelength": None,
    #     "full_width_half_max": None,
    # }

    # CLOUD_MASK = CLOUD_MASK.assign_coords(band_coords).transpose(
    #     "time", "band", "y", "x"
    # )
    
    arr = arr.sel(time = arr.time.dt.date.isin(CLOUD_MASK.time.dt.date.values))
    CLOUD_MASK = CLOUD_MASK.sel(time = CLOUD_MASK.time.dt.date.isin(arr.time.dt.date.values))

    dupls = pd.Index(arr.time).duplicated()
    if dupls.sum() != 0:
        arr = arr.groupby("time").median()
    dupls = pd.Index(CLOUD_MASK.time).duplicated()
    if dupls.sum() != 0:
        CLOUD_MASK = CLOUD_MASK.groupby("time").reduce(cloud_mask_reduce, dim = "time")#.max()
    # try:
    #     done = False
    #     while not done:
    #         dupls = pd.Index(arr.time).duplicated()
    #         if dupls.sum() == 0:
    #             done = True
    #         else:
    #             arr = arr.groupby("time").median()
    #             # new_time = arr.time.values
    #             # new_time[dupls] += np.timedelta64(1, 'm')
    #             # arr["time"] = new_time
    #     done = False
    #     while not done:
    #         dupls = pd.Index(CLOUD_MASK.time).duplicated()
    #         if dupls.sum() == 0:
    #             done = True
    #         else:
    #             CLOUD_MASK = CLOUD_MASK.groupby("time").median()
    #             # new_time = CLOUD_MASK.time.values
    #             # new_time[dupls] += np.timedelta64(1, 'm')
    #             # CLOUD_MASK["time"] = new_time
    # except ValueError:
    #     breakpoint()
    # available_times = CLOUD_MASK.time.data
    # original_times = arr.time.data
    # tmp_times = original_times.astype("datetime64[D]").astype("datetime64[ns]")
    # arr["time"] = tmp_times
    # arr = arr.loc[tmp_times[np.isin(tmp_times, available_times)]]
    # arr["time"] = original_times[np.isin(tmp_times, available_times)]
    # CLOUD_MASK = CLOUD_MASK.loc[available_times[np.isin(available_times, tmp_times)]]

    # CLOUD_MASK = CLOUD_MASK.assign_coords(time=("time", arr.time.data))

    s2_cloudmask = xr.concat([arr, CLOUD_MASK], dim="band", coords = "minimal", compat = "override", combine_attrs = "override")

    # try:
    #     s2_cloudmask = xr.concat([arr, CLOUD_MASK], dim="band", coords = "minimal", compat = "override", combine_attrs = "override")
    # except IndexError:
    #     breakpoint()
    # except TypeError:
    #     breakpoint()
    # except ValueError:
    #     breakpoint()

    return s2_cloudmask


# def filterImages(arr, percentage=0.1):
#     """Filters the array.

#     Parameters
#     ----------
#     arr : xarray.DataArray
#         Stacked array with the cloud mask.
#     percentage : float, default = 0.1
#         Percentage of valid pixels. Keep everything greater than this.

#     Returns
#     -------
#     xarray.DataArray
#         Filtered stacked array.
#     """
#     valid_pixels = arr.sel(band="CLOUD_MASK") == 0
#     valid_percentage = valid_pixels.mean(["x", "y"])
#     valid_pixels = valid_pixels.assign_coords({"band": "VALID"})
#     S2_VALID = xr.concat([arr, valid_pixels], dim="band")
#     S2_VALID = S2_VALID.assign_coords(
#         {"valid_percentage": ("time", valid_percentage.values)}
#     )
#     S2_VALID = S2_VALID.where(S2_VALID.valid_percentage > percentage, drop=True)

#     return S2_VALID


# def sen2flux_core(catalog, aoi, bbox, cloud_percentage, iniDate, endDate, year, epsg):
#     """Core code for creating a sen2flux cube.

#     Parameters
#     ----------
#     catalog : pystac.Catalog
#         Catalog to sign.
#     aoi : dict
#         GeoJSON of the Bounding Box in geographic coordinates.
#     bbox : dict
#         GeoJSON of the Bounding Box.
#     cloud_percentage : float
#         Filter STAC by this value.
#     iniDate : string
#         Initial date to filter the STAC in YYYY-MM-DD.
#     endDate : string
#         Final date to filter the STAC in YYYY-MM-DD.
#     year : int
#         Year to look for in GEE.
#     epsg : int
#         CRS.

#     Returns
#     -------
#     xarray.DataArray
#         sen2flux cube.
#     """
#     SEARCH = catalog.search(
#         intersects=aoi,
#         datetime=f"{iniDate}/{endDate}",
#         collections=["sentinel-2-l2a"],
#         query={"eo:cloud_cover": {"lt": cloud_percentage}},
#     )

#     print("Resolving Planetary Computer signatures and stacking...")

#     items = list(SEARCH.get_items())
#     REFERENCE = signAndStack(items, bbox, epsg)

#     print("Adding Sun and View Angles...")

#     SVA, REFERENCE = sunAndViewAngles(items, REFERENCE)

#     print("Computing NBAR...")

#     NBAR_ARR = computeNBAR(SVA, REFERENCE)

#     print("Computing Cloud Mask from Google Earth Engine...")

#     NBAR_CLOUDMASK = computeCloudMask(aoi, NBAR_ARR, REFERENCE, year)

#     print("Filtering stack by valid pixels...")

#     FILTERED = filterImages(NBAR_CLOUDMASK, 0.00625)

#     return FILTERED


# def sen2flux(coords, buffer, year):
#     """Creates a cube for a single year.

#     Parameters
#     ----------
#     coords : list
#         Geographic coordinates.
#     buffer : int | float
#         Buffer distance to create the cube in meters.
#     year : int
#         Year to look for.

#     Returns
#     -------
#     xarray.DataArray
#         sen2flux cube for a specific year.
#     """
#     s1iDate = f"{year}-01-01"
#     s1eDate = f"{year}-06-30"

#     s2iDate = f"{year}-07-01"
#     s2eDate = f"{year}-12-31"

#     print(f"Resolving Bounding Box for coordinates {coords}...")

#     utm_crs_list = query_utm_crs_info(
#         datum_name="WGS 84",
#         area_of_interest=AreaOfInterest(coords[0], coords[1], coords[0], coords[1]),
#     )

#     epsg = int(utm_crs_list[0].code)

#     aoi = towerFootprint(coords[0], coords[1], buffer)
#     bbox = rasterio.features.bounds(towerFootprint(coords[0], coords[1], buffer, False))

#     print(f"Resolving Sentinel-2 L2A for year {year}...")

#     CATALOG = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1"
#     )
#     valid_area = (1 - (0.5 * ((buffer * 2 / 10) ** 2) / (10980**2))) * 100

#     print("Resolving 1st semester...")

#     to_concat = []

#     try:
#         S2S1 = sen2flux_core(
#             CATALOG, aoi, bbox, valid_area, s1iDate, s1eDate, year, epsg
#         ).compute()
#         S2S1.name = "sen2flux"
#         to_concat.append(S2S1)
#     except:
#         print("No images found for the first semester!")

#     print("Resolving 2nd semester...")

#     try:
#         S2S2 = sen2flux_core(
#             CATALOG, aoi, bbox, valid_area, s2iDate, s2eDate, year, epsg
#         ).compute()
#         S2S2.name = "sen2flux"
#         to_concat.append(S2S2)
#     except:
#         print("No images found for the second semester!")

#     print("Concatenating and adding attributes...")

#     if len(to_concat) > 0:
#         # S2FLX = xr.concat([S2S1,S2S2],dim = "time")

#         if len(to_concat) > 1:
#             S2FLX = xr.concat(to_concat, dim="time")
#         else:
#             S2FLX = to_concat[0]

#         S2FLX.attrs["crs"] = f"+init=epsg:{S2FLX.epsg.data}"
#         S2FLX.attrs["res"] = np.array([10.0, 10.0])
#         S2FLX.attrs["coords"] = np.array(coords)
#         S2FLX.attrs["xy"] = np.array(towerCoordinates(coords[0], coords[1]))
#         S2FLX.attrs["scales"] = np.concatenate(
#             (np.repeat(0.0001, 12), 1.0, np.repeat(0.0001, 9), 1.0, 1.0), axis=None
#         )
#         S2FLX.attrs["offsets"] = np.repeat(0.0, 24)

#         S2FLX = S2FLX.reset_coords(drop=True)

#         if "spec" in S2FLX.attrs.keys():
#             del S2FLX.attrs["spec"]
#             del S2FLX.attrs["transform"]
#             del S2FLX.attrs["resolution"]

#         print("Done!")

#         return S2FLX
#     else:
#         print(f"No images were found for {year}")


# def sen2flux_semester(coords, buffer, year, semester):

#     if semester == 1:

#         s1iDate = f"{year}-01-01"
#         s1eDate = f"{year}-03-31"

#         s2iDate = f"{year}-04-01"
#         s2eDate = f"{year}-06-30"

#     if semester == 2:

#         s1iDate = f"{year}-07-01"
#         s1eDate = f"{year}-09-30"

#         s2iDate = f"{year}-10-01"
#         s2eDate = f"{year}-12-31"

#     print(f"Resolving Bounding Box for coordinates {coords}...")

#     utm_crs_list = query_utm_crs_info(
#         datum_name="WGS 84",
#         area_of_interest=AreaOfInterest(coords[0], coords[1], coords[0], coords[1]),
#     )

#     epsg = int(utm_crs_list[0].code)

#     aoi = towerFootprint(coords[0], coords[1], buffer)
#     bbox = rasterio.features.bounds(towerFootprint(coords[0], coords[1], buffer, False))

#     print(f"Resolving Sentinel-2 L2A for year {year}-{semester}")

#     CATALOG = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1"
#     )
#     valid_area = (1 - (0.5 * ((buffer * 2 / 10) ** 2) / (10980**2))) * 100

#     print("Resolving 1st part...")

#     to_concat = []

#     try:
#         S2S1 = sen2flux_core(
#             CATALOG, aoi, bbox, valid_area, s1iDate, s1eDate, year, epsg
#         ).compute()
#         S2S1.name = "sen2flux"
#         to_concat.append(S2S1)
#     except:
#         print("No images found for the first part!")

#     print("Resolving 2nd part...")

#     try:
#         S2S2 = sen2flux_core(
#             CATALOG, aoi, bbox, valid_area, s2iDate, s2eDate, year, epsg
#         ).compute()
#         S2S2.name = "sen2flux"
#         to_concat.append(S2S2)
#     except:
#         print("No images found for the second part!")

#     print("Concatenating and adding attributes...")

#     if len(to_concat) > 0:
#         # S2FLX = xr.concat([S2S1,S2S2],dim = "time")

#         if len(to_concat) > 1:
#             S2FLX = xr.concat(to_concat, dim="time")
#         else:
#             S2FLX = to_concat[0]

#         S2FLX.attrs["crs"] = f"+init=epsg:{S2FLX.epsg.data}"
#         S2FLX.attrs["res"] = np.array([10.0, 10.0])
#         S2FLX.attrs["coords"] = np.array(coords)
#         S2FLX.attrs["xy"] = np.array(towerCoordinates(coords[0], coords[1]))
#         S2FLX.attrs["scales"] = np.concatenate(
#             (np.repeat(0.0001, 12), 1.0, np.repeat(0.0001, 9), 1.0, 1.0), axis=None
#         )
#         S2FLX.attrs["offsets"] = np.repeat(0.0, 24)

#         S2FLX = S2FLX.reset_coords(drop=True)

#         if "spec" in S2FLX.attrs.keys():
#             del S2FLX.attrs["spec"]
#             del S2FLX.attrs["transform"]
#             del S2FLX.attrs["resolution"]

#         print("Done!")

#         return S2FLX
#     else:
#         print(f"No images were found for {year}-{semester}")


# def sen2flux_month(coords, buffer, year):

#     print(f"Resolving Bounding Box for coordinates {coords}...")

#     utm_crs_list = query_utm_crs_info(
#         datum_name="WGS 84",
#         area_of_interest=AreaOfInterest(coords[0], coords[1], coords[0], coords[1]),
#     )

#     epsg = int(utm_crs_list[0].code)

#     aoi = towerFootprint(coords[0], coords[1], buffer)
#     bbox = rasterio.features.bounds(towerFootprint(coords[0], coords[1], buffer, False))

#     print(f"Resolving Sentinel-2 L2A for year {year}")

#     CATALOG = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1"
#     )
#     # valid_area = (1 - (0.5 * ((buffer * 2 / 10) ** 2) / (10980 ** 2))) * 100
#     valid_area = (1 - (0.5 * ((640 * 2 / 10) ** 2) / (10980**2))) * 100

#     to_concat = []

#     print("Resolving 1st part...")

#     for month in [
#         "01",
#         "02",
#         "03",
#         "04",
#         "05",
#         "06",
#         "07",
#         "08",
#         "09",
#         "10",
#         "11",
#         "12",
#     ]:

#         try:
#             S2S1 = sen2flux_core(
#                 CATALOG,
#                 aoi,
#                 bbox,
#                 valid_area,
#                 f"{year}-{month}-01",
#                 f"{year}-{month}-{monthrange(year,int(month))[1]}",
#                 year,
#                 epsg,
#             ).compute()
#             S2S1.name = "sen2flux"
#             to_concat.append(S2S1)
#         except:
#             print(f"No images found for month {year}-{month}!")

#     print("Concatenating and adding attributes...")

#     if len(to_concat) > 0:
#         # S2FLX = xr.concat([S2S1,S2S2],dim = "time")

#         if len(to_concat) > 1:
#             S2FLX = xr.concat(
#                 to_concat, dim="time", coords="minimal", compat="override"
#             )
#         else:
#             S2FLX = to_concat[0]

#         S2FLX.attrs["crs"] = f"+init=epsg:{S2FLX.epsg.data}"
#         S2FLX.attrs["res"] = np.array([10.0, 10.0])
#         S2FLX.attrs["coords"] = np.array(coords)
#         S2FLX.attrs["xy"] = np.array(towerCoordinates(coords[0], coords[1]))
#         S2FLX.attrs["scales"] = np.concatenate(
#             (np.repeat(0.0001, 12), 1.0, np.repeat(0.0001, 9), 1.0, 1.0), axis=None
#         )
#         S2FLX.attrs["offsets"] = np.repeat(0.0, 24)

#         S2FLX = S2FLX.reset_coords(drop=True)

#         if "spec" in S2FLX.attrs.keys():
#             del S2FLX.attrs["spec"]
#             del S2FLX.attrs["transform"]
#             del S2FLX.attrs["resolution"]

#         print("Done!")

#         return S2FLX
#     else:
#         print(f"No images were found for {year}")


# def sen2flux_multi(coords, buffer, ini_year, end_year):

#     years = list(range(ini_year, end_year + 1))
#     DCS = []
#     for year in years:
#         DCS.append(sen2flux(coords, buffer, year))

#     S2FLX = xr.concat(DCS, dim="time")

#     return S2FLX
