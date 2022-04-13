
import argparse
import cdsapi

import xarray as xr
import numpy as np

import zarr
from numcodecs import Blosc
from tqdm import tqdm

from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download 3-hourly ERA5 NetCDFs over Africa.')
    parser.add_argument('outpath', type=str, help='directory where to save downloads')
    args = parser.parse_args()

    c = cdsapi.Client()

    SHORT_NAMES = {
        't2m': '2m_temperature', 
        'pev': 'potential_evaporation', 
        'slhf': 'surface_latent_heat_flux',
        'ssr': 'surface_net_solar_radiation', 
        'sp': 'surface_pressure', 
        'sshf': 'surface_sensible_heat_flux',
        'e': 'total_evaporation', 
        'tp': 'total_precipitation'
    }
    SHORT_NAMES_INV = {v: k for k,v in SHORT_NAMES.items()}

    variables = ['2m_temperature', 'potential_evaporation', 'surface_latent_heat_flux',
                'surface_net_solar_radiation', 'surface_pressure', 'surface_sensible_heat_flux',
                'total_evaporation', 'total_precipitation']

    for variable in variables:
        print(f"Downloading {variable}")
        c.retrieve(
        'reanalysis-era5-land',
        {
            'format': 'netcdf',
            'year': [
                '2017', '2018', '2019',
                '2020', '2021', '2022',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', 
                '03:00',
                '06:00', 
                '09:00', 
                '12:00', 
                '15:00', 
                '18:00',
                '21:00'
            ],
            'variable': variable,
            'area': [
                40, -25, -40,
                55,
            ],
        },
        str(Path(args.outpath)/f'era5_{SHORT_NAMES_INV[variable]}.nc'))

    era5 = xr.merge([xr.open_dataset(ncpath, chunks = {"latitude": 20, "longitude": 20, "time": 8*365}).rename({"latitude": "lat", "longitude": "lon"}) for ncpath in Path(args.outpath).glob("*.nc")])

    ds = xr.Dataset(coords = dict(era5.coords))
    
    ds = ds.chunk(chunks={"time": 8*365, "lat": 20, "lon": 20})

    zarrpath = str(Path(args.outpath)/"era5_africa_0d1_3hourly.zarr")

    ds.to_zarr(zarrpath)

    zarrgroup = zarr.open_group(zarrpath)

    compressor = Blosc(cname='lz4', clevel=1)

    for var in list(SHORT_NAMES.keys()):
        newds = zarrgroup.create_dataset(var, shape = (len(era5.time.values),len(era5.lat.values), len(era5.lon.values)), chunks = (8*365, 20, 20), dtype = 'float32', fillvalue = np.nan, compressor = compressor)
        newds.attrs['_ARRAY_DIMENSIONS'] = ("time", "lat", "lon")


    for var in tqdm(list(SHORT_NAMES.keys())):
        
        era5var = xr.open_dataset(str(Path(args.outpath)/f"era5_{var}.nc"))

        zarrgroup[var][:] = era5var[var].values
    