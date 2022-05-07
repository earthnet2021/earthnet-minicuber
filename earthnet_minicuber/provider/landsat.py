

import collections
import os
import pystac_client
import stackstac
import rasterio
import numpy as np
import xarray as xr

import odc.algo

from . import provider_base



def set_value_at_index(bitmask, index, value):
    # Taken from https://github.com/opendatacube/datacube-core/blob/develop/datacube/utils/masking.py
    # This file is part of the Open Data Cube, see https://opendatacube.org for more information
    #
    # Copyright (c) 2015-2020 ODC Contributors
    # SPDX-License-Identifier: Apache-2.0
    """
    Set a bit value onto an integer bitmask
    eg. set bits 2 and 4 to True
    >>> mask = 0
    >>> mask = set_value_at_index(mask, 2, True)
    >>> mask = set_value_at_index(mask, 4, True)
    >>> print(bin(mask))
    0b10100
    >>> mask = set_value_at_index(mask, 2, False)
    >>> print(bin(mask))
    0b10000
    :param bitmask: existing int bitmask to alter
    :type bitmask: int
    :type index: int
    :type value: bool
    """
    bit_val = 2 ** index
    if value:
        bitmask |= bit_val
    else:
        bitmask &= (~bit_val)
    return bitmask

def create_mask_value(bits_def, **flags):
    # Taken from https://github.com/opendatacube/datacube-core/blob/develop/datacube/utils/masking.py
    # This file is part of the Open Data Cube, see https://opendatacube.org for more information
    #
    # Copyright (c) 2015-2020 ODC Contributors
    # SPDX-License-Identifier: Apache-2.0
    mask = 0
    value = 0

    for flag_name, flag_ref in flags.items():
        defn = bits_def.get(flag_name, None)
        if defn is None:
            raise ValueError(f'Unknown flag: "{flag_name}"')

        try:
            [flag_value] = (bit_val
                            for bit_val, val_ref in defn['values'].items()
                            if val_ref == flag_ref)
            flag_value = int(flag_value)  # Might be string if coming from DB
        except ValueError:
            raise ValueError('Unknown value %s specified for flag %s' %
                             (flag_ref, flag_name))

        if isinstance(defn['bits'], collections.abc.Iterable):  # Multi-bit flag
            # Set mask
            for bit in defn['bits']:
                mask = set_value_at_index(mask, bit, True)

            shift = min(defn['bits'])
            real_val = flag_value << shift

            value |= real_val

        else:
            bit = defn['bits']
            mask = set_value_at_index(mask, bit, True)
            value = set_value_at_index(value, bit, flag_value)

    return mask, value
    

class Landsat(provider_base.Provider):

    SENSORS = ["ls5_st", "ls5_sr", "ls7_st", "ls7_sr", "ls8_st", "ls8_sr", "ls9_st", "ls9_sr"]
    SPECTRAL_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"]
    THERMAL_BANDS = ["ST_B6", "ST_B10", "QA_PIXEL"]

    DESCRIPTIONS_57_sr = {
        'SR_B1': 'Surface reflectance band 1 (Blue)',
        'SR_B2': 'Surface reflectance band 2 (Green)',
        'SR_B3': 'Surface reflectance band 3 (Red)',
        'SR_B4': 'Surface reflectance band 4 (Near-Infrared (NIR))',
        'SR_B5': 'Surface reflectance band 5 (Short Wavelength Infrared (SWIR) 1)',
        'SR_B7': 'Surface reflectance band 7 (SWIR 2)',
        'QA_PIXEL': 'Pixel quality',
        'QA_RADSAT': 'Radiometric saturation',
        'SR_ATMOS _OPACITY': 'Atmospheric opacity',
        'SR_CLOUD _QA': 'Cloud mask quality'
    }

    DESCRIPTIONS_57_st = {
        'ST_B6': 'Surface temperature band 6 (Thermal Infrared (TIR))',
        'ST_TRAD': 'Thermal radiance',
        'ST_URAD': 'Upwell radiance',
        'ST_DRAD': 'Downwell radiance',
        'ST_ATRAN': 'Atmospheric transmittance',
        'ST_EMIS': 'Emissivity',
        'ST_EMSD': 'Emissivity standard deviation',
        'ST_CDIST': 'Distance to cloud',
        'QA_PIXEL': 'Pixel quality',
        'QA_RADSAT': 'Radiometric saturation',
        'ST_QA': 'Surface temperature uncertainty'
    }
    DESCRIPTIONS_89_sr = {
        'SR_B1': 'Surface reflectance band 1 (Coastal Aerosol)',
        'SR_B2': 'Surface reflectance band 2 (Blue)',
        'SR_B3': 'Surface reflectance band 3 (Green)',
        'SR_B4': 'Surface reflectance band 4 (Red)',
        'SR_B5': 'Surface reflectance band 5 (NIR)',
        'SR_B6': 'Surface reflectance band 6 (SWIR 1)',
        'SR_B7': 'Surface reflectance band 7 (SWIR 2)',
        'QA_PIXEL': 'Pixel quality',
        'QA_RADSAT': 'Radiometric saturation',
        'SR_QA _AEROSOL': 'Aerosol level'
    }

    DESCRIPTIONS_89_st = {
        'ST_B10': 'Surface temperature band 10 (TIR)',
        'ST_TRAD': 'Thermal radiance',
        'ST_URAD': 'Upwell radiance',
        'ST_DRAD': 'Downwell radiance',
        'ST_ATRAN': 'Atmospheric transmittance',
        'ST_EMIS': 'Emissivity',
        'ST_EMSD': 'Emissivity standard deviation',
        'ST_CDIST': 'Distance to cloud',
        'QA_PIXEL': 'Pixel quality',
        'QA_RADSAT': 'Radiometric saturation',
        'ST_QA': 'Surface temperature uncertainty'
    }

    DESCRIPTIONS_BY_SENSOR = {
        "ls5_st": DESCRIPTIONS_57_st,
        "ls5_sr": DESCRIPTIONS_57_sr,
        "ls7_st": DESCRIPTIONS_57_st,
        "ls7_sr": DESCRIPTIONS_57_sr,
        "ls8_st": DESCRIPTIONS_89_st,
        "ls8_sr": DESCRIPTIONS_89_sr,
        "ls9_st": DESCRIPTIONS_89_st,
        "ls9_sr": DESCRIPTIONS_89_sr
    }

    PIXELQ_FLAGS = {
                    "cirrus": {
                        "bits": 2,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "cirrus_confidence": {
                        "bits": [14, 15],
                        "values": {"0": "none", "1": "low", "2": "reserved", "3": "high"},
                    },
                    "clear": {"bits": 6, "values": {"0": False, "1": True}},
                    "cloud": {
                        "bits": 3,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "cloud_confidence": {
                        "bits": [8, 9],
                        "values": {"0": "none", "1": "low", "2": "medium", "3": "high"},
                    },
                    "cloud_shadow": {
                        "bits": 4,
                        "values": {"0": "not_high_confidence", "1": "high_confidence"},
                    },
                    "cloud_shadow_confidence": {
                        "bits": [10, 11],
                        "values": {"0": "none", "1": "low", "2": "reserved", "3": "high"},
                    },
                    "dilated_cloud": {"bits": 1, "values": {"0": "not_dilated", "1": "dilated"}},
                    "nodata": {"bits": 0, "values": {"0": False, "1": True}},
                    "snow": {"bits": 5, "values": {"0": "not_high_confidence", "1": "high_confidence"}},
                    "snow_ice_confidence": {
                        "bits": [12, 13],
                        "values": {"0": "none", "1": "low", "2": "reserved", "3": "high"},
                    },
                    "water": {"bits": 7, "values": {"0": "land_or_cloud", "1": "water"}},
                }


    def __init__(self, sensor = "ls8_sr", bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], cloud_mask = True, mask_kwargs = {"cloud": "high_confidence", # True where there is cloud
                #cirrus="high_confidence",# True where there is cirrus cloud
                "cloud_shadow":"high_confidence",# True where there is cloud shadow
                "dilated_cloud": "dilated",
                "nodata": True}):

        self.sensor = sensor
        if cloud_mask and ("QA_PIXEL" not in bands):
            bands.append("QA_PIXEL")
            self.drop_qa = True
        else:
            self.drop_qa = False
        self.bands = bands
        self.cloud_mask = cloud_mask
        self.mask_kwargs = mask_kwargs

        URL = "https://explorer.digitalearth.africa/stac/"
        self.catalog = pystac_client.Client.open(URL)

        os.environ['AWS_NO_SIGN_REQUEST'] = "TRUE"
        os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'


    def load_data(self, bbox, time_interval, **kwargs):
        
        with rasterio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com'):
            items_ls = self.catalog.search(
                    bbox = bbox,
                    collections=[self.sensor],
                    datetime=time_interval
                ).get_all_items()
            
            if len(items_ls.to_dict()['features']) == 0:
                return None
                
            metadata = items_ls.to_dict()['features'][0]["properties"]
            epsg = metadata["proj:epsg"]

            stack = stackstac.stack(items_ls, epsg = epsg, assets = self.bands, dtype = "float32", properties = False, band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 1024)


            ls_bands = [f"{self.sensor}_{b.split('_')[1] if b!= 'QA_PIXEL' else b}" for b in stack.band.values]
            stack["band"] = ls_bands

            stack = stack.to_dataset("band")

            if self.cloud_mask:
                mask, _ = create_mask_value(
                    self.PIXELQ_FLAGS, **self.mask_kwargs
                )
                
                pq_mask = (stack[f"{self.sensor}_QA_PIXEL"].astype("uint16") & mask) != 0

                pq_mask = odc.algo.mask_cleanup(pq_mask, mask_filters=[("opening", 4),("dilation", 6)])

                stack[f"{self.sensor}_mask"] = pq_mask.astype("uint8")

            for b in ls_bands:
                if b != f"{self.sensor}_QA_PIXEL":
                    if self.sensor.endswith("st"):
                        stack[b] = (0.00341802 * stack[b] + 149.0).astype("float32")
                    else:
                        stack[b] = (2.75e-05 * stack[b] - 0.2).astype("float32")
            
            

            if self.cloud_mask:
                cloud_mask = stack[f"{self.sensor}_mask"].groupby("time.date").max("time").rename({"date": "time"})

            stack = stack.groupby("time.date").median("time").rename({"date": "time"})

            if self.cloud_mask:
                stack[f"{self.sensor}_mask"] = cloud_mask
                stack[f"{self.sensor}_mask"].attrs = {"provider": f"Landsat {self.sensor[2]} {self.sensor[4:].upper()}", "interpolation_type": "nearest", "description": "Data mask", "classes": """
                0 - Valid
                1 - Invalid
                """}


            for b in self.bands:
                bandname = f"{self.sensor}_{b.split('_')[1] if b!= 'QA_PIXEL' else b}"
                if bandname in ls_bands:
                    stack[bandname].attrs = {"provider": f"Landsat {self.sensor[2]} {self.sensor[4:].upper()}", "interpolation_type": "linear" if b != f"{self.sensor}_QA_PIXEL" else "nearest", "description": self.DESCRIPTIONS_BY_SENSOR[self.sensor][b]}


            if self.drop_qa:
                stack = stack.drop_vars([f"{self.sensor}_QA_PIXEL"], errors = "ignore")

            stack = stack.drop_vars(["epsg", "id"], errors = "ignore")
            
            stack["time"] = np.array([str(d) for d in stack.time.values], dtype="datetime64[D]")

            stack.attrs["epsg"] = epsg

            return stack
