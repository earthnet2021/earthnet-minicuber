
import argparse
import numpy as np

import earthnet_minicuber as emc
import pandas as pd
from pathlib import Path
from copy import deepcopy


awsspec = {
            "lon_lat": (43.598946, 3.087414), # center pixel
            "xy_shape": (128,128), # width, height of cutout around center pixel
            "resolution": 30, # in meters.. will use this together with grid of primary provider..
            "time_interval": "2018-01-01/2021-12-31",
            "providers": [
                {
                    "name": "s2",
                    "kwargs": {"bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"],#, "B09", "B11", "B12"], 
                    "best_orbit_filter": True, "brdf_correction": True, "cloud_mask": False, "aws_bucket": "element84"}
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
                }
            ]
    }

localspec = {
            "lon_lat": (43.598946, 3.087414), # center pixel
            "xy_shape": (128,128), # width, height of cutout around center pixel
            "resolution": 30, # in meters.. will use this together with grid of primary provider..
            "time_interval": "2018-01-01/2021-12-31",
            "providers": [
                {
                    "name": "s2",
                    "kwargs": {"bands": ["SCL"],#, "B09", "B11", "B12"], 
                    "best_orbit_filter": True, "brdf_correction": False, "cloud_mask": True, "aws_bucket": "element84"}
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('idx', type=int, help='index of the minicube')
    parser.add_argument('type', type=str, help='aws or local')
    parser.add_argument('dataset', type=str, help='train or test')
    parser.add_argument('csvpath', type = str, help = "path to csv")
    parser.add_argument('savepath', type = str, help = "savepath")

    

    args = parser.parse_args()

    basespec = awsspec if args.type == "aws" else localspec

    df = pd.read_csv(args.csvpath)#"/Net/Groups/BGI/scratch/DeepCube/UC1/sampled_minicubes_v2_200000.csv"

    df = df[df.set == args.dataset]

    row = df.iloc[args.idx]

    lon_lat_center = ((row["MinLon"] + row["MaxLon"])/2, (row["MinLat"] + row["MaxLat"])/2)

    spec = deepcopy(basespec)
    spec["lon_lat"] = lon_lat_center

    start_date = np.datetime64('2018-01-01') + np.random.default_rng(args.idx).integers(0, 157) if args.dataset == "train" else np.datetime64('2018-01-01')

    end_date = start_date + 460 if args.dataset == "train" else np.datetime64('2022-03-31')

    spec["time_interval"] = f"{start_date}/{end_date}"


    name = row["CubeID"]
    country = row["country"].replace(" ", "")

    savepath = Path(args.savepath)

    savepath_cube = savepath/args.dataset/country/f"{name}_{args.type}.nc"

    pars = {
        "specs": spec,
        "savepath": savepath_cube,
        "verbose": True
    }

    print(f"Working on {name} at {lon_lat_center} in {country} from {start_date} to {end_date}.")

    emc.Minicuber.save_minicube_mp(pars)

