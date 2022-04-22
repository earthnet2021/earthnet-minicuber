import yaml
import argparse
import cdsapi


import xarray as xr
import numpy as np

import zarr
from numcodecs import Blosc
from tqdm import tqdm

from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download daily ECMWF Seasonal Forecasts GRIB over Africa.')
    parser.add_argument('outpath', type=str, help='directory where to save downloads')
    args = parser.parse_args()

    leadtime_range = list(range(0,24*180,6)) # every 6 hours for 180 days

    variables = ['2m_temperature', 'surface_latent_heat_flux',
                 'surface_net_solar_radiation','mean_sea_level_pressure', 'evaporation',
                'surface_sensible_heat_flux','total_precipitation']
    
    subdaily_vars = ['2m_temperature', 'mean_sea_level_pressure']
    
    SHORT_NAMES = {
        't2m': '2m_temperature', 
        'slhf': 'surface_latent_heat_flux',
        'ssr': 'surface_net_solar_radiation', 
        'mslp': 'mean_sea_level_pressure', 
        'sshf': 'surface_sensible_heat_flux',
        'e': 'evaporation', 
        'tp': 'total_precipitation',
    }
    
    years = ['2017', '2018', '2019','2020','2021']

    for year in years:
        for variable in variables:

            file_path = Path(args.outpath+f'ecmwf_{year}_{variable}.grib')
            if file_path.is_file():
                print(f"Skipping ecmwf_{year}_{variable}.grib, file found!")
                continue

            print(f"Downloading ecmwf_{year}_{variable}.grib")
            try:
                c.retrieve(
                    'seasonal-original-single-levels',
                    {
                        'format': 'grib',
                        'originating_centre': 'ecmwf',
                        'system': '5',
                        'variable': str(variable),
                        'year': str(year),
                        'month': [
                            '01', '02', '03',
                            '04', '05', '06',
                            '07', '08', '09',
                            '10', '11', '12',
                            ],
                        'day': '01',
                        'leadtime_hour': leadtime_range,
                        'area': [ 40, -25, -40, 55,], # Coarsely Africa
                    },
                    file_path)
            except:
                pass

#    try:
#        if Path(args.outpath+'coords.yml').exists():
#            print("Coords YAML found!")
#            print("Trying to load coords from YAML")
#        with open(args.outpath+'coords.yml', 'r') as stream:
#            coords=yaml.safe_load(stream)
#            print(coords)
#            print("Coords loaded from YAML")
#    except:
#        print("Cannot load coords from YAML. Starting xr.merge")
    
    for year_idx in range(len(years)):
        print(f"Starting year {years[year_idx]}")
        ecmwf = xr.merge([xr.open_dataset(gribpath, chunks = {"number":1,"time":1,"latitude": 20, "longitude": 20, "time": 4*180}).rename({"latitude": "lat", "longitude": "lon"}) for gribpath in Path(args.outpath).glob(f"*{years[year_idx]}_2m_temperature.grib")])
    #        print("xr.merge finished")
    #        print(dict(ecmwf.coords))

    #        print("Saving coords to YAML")
    #        with open(args.outpath+'coords.yml', 'w') as outfile:
    #            yaml.dump(dict(ecmwf.coords), outfile, default_flow_style=False)

        coords = dict(ecmwf.coords)

        ds = xr.Dataset(coords = coords)
        ds = ds.chunk(chunks={"step":4*180, "number":1,"time": 1, "lat": 20, "lon": 20})

        print("Creating path")
        zarrpath = str(Path(args.outpath)/f"ecmwf_africa_{years[year_idx]}_1d0_6hourly.zarr")
        ds.to_zarr(zarrpath)

        zarrgroup = zarr.open_group(zarrpath)
        compressor = Blosc(cname='lz4', clevel=1)

        for var in list(SHORT_NAMES.keys()):
            print(f"Creating dataset for {var}")
            newds = zarrgroup.create_dataset(var, 
                                             shape = (51, 12*len(years), 4*180-1, 81, 81), 
                                             chunks = (1, 1, 4*180,20, 20), 
                                             dtype = 'float32', 
                                             fillvalue = np.nan,
                                             compressor = compressor
                                            )

            newds.attrs['_ARRAY_DIMENSIONS'] = ("number","time", "step","lat", "lon")

    
        for var in tqdm((SHORT_NAMES.keys())):
            print(f"Creating Zarr files year {years[year_idx]} variable {var}")
            if SHORT_NAMES[var] in subdaily_vars:
                idxs = list(range(0,180*4-1,1))
            else:
                idxs = list(range(-1,180*4-1,4))[1:]
            
            #step_mask = np.isin(np.arange(180), idxs)
            yearvar = xr.open_dataset(str(Path(args.outpath)/f"ecmwf_{years[year_idx]}_{SHORT_NAMES[var]}.grib"))
            
            array = np.empty((51,12,4*180,81,81))
            array[:] = np.NaN
            
            array[:,year_idx*12:year_idx*12+12,idxs,:,:] = yearvar[var].values
            zarrgroup[var] = array
