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

try: 
    import ee
    import wxee
except ImportError: 
    ee = None
    wxee = None
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio.features
import stackstac
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
import random
import shutil


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
                if aws_bucket == "planetary_computer":
                    print(f"No BRDF metadata found for {item.id}. Thus skipping.")
                else:
                    print(f"No BRDF metadata found for {item.properties['sentinel:product_id']}. Thus skipping.")

            iC = iC + 1
    ref = ref[idx]

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
        )
    
    if len(angles_array) == 0:
        return None, None
    md = xr.concat(angles_array, dim="time")
    md = md.assign_coords(time=("time", ref.time.data))


    complete = xr.concat([ref, md], dim="band")

    return complete, ref



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

    if ee:
        ee_aoi = ee.Geometry(aoi)
    else:
        return None

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

    for attempt in range(30):
        try:
            CLOUD_MASK_xarray = CLOUD_MASK.wx.to_xarray(
                scale=10, crs="EPSG:" + str(arr.attrs["epsg"]), region=ee_aoi, progress = False,num_cores = 2,max_attempts=2
            )
        except (requests.exceptions.HTTPError, ee.ee_exception.EEException):
            sleeptime = random.uniform(5,30)
            print(f"Earth engine overload... sleeping {sleeptime:.2f} sec. AOI: {aoi['coordinates'][0][0]}")
            time.sleep(sleeptime)
        except OSError as err:
            try:
                shutil.rmtree(str(err).split(" ")[-1])
            except:
                print(err, str(err).split(" ")[-1])
            time.sleep(random.uniform(0,4))
        else:
            break
    else:
        print(f"No Cloud mask for Sentinel IDs: {arr.id.data}")
        return arr
    
    CLOUD_MASK_xarray = CLOUD_MASK_xarray.where(lambda x: x >= 0, other=4)

    CLOUD_MASK = CLOUD_MASK_xarray.CLOUD_MASK.interp(
            x=arr.x.data,
            y=arr.y.data,
            method="nearest",
            kwargs={"fill_value": 4},
        )
    
    
    CLOUD_MASK = CLOUD_MASK.assign_coords(band="mask").expand_dims("band")

    
    arr = arr.sel(time = arr.time.dt.date.isin(CLOUD_MASK.time.dt.date.values))
    CLOUD_MASK = CLOUD_MASK.sel(time = CLOUD_MASK.time.dt.date.isin(arr.time.dt.date.values))

    dupls = pd.Index(arr.time).duplicated()
    if dupls.sum() != 0:
        arr = arr.groupby("time").median()
    dupls = pd.Index(CLOUD_MASK.time).duplicated()
    if dupls.sum() != 0:
        CLOUD_MASK = CLOUD_MASK.groupby("time").reduce(cloud_mask_reduce, dim = "time")#.max()

    s2_cloudmask = xr.concat([arr, CLOUD_MASK], dim="band", coords = "minimal", compat = "override", combine_attrs = "override")

    s2_cloudmask = s2_cloudmask.drop_vars(["id", "id_old", "sentinel:data_coverage", "sentinel:sequence"], errors = "ignore")

    return s2_cloudmask
