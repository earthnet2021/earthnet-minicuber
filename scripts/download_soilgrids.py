
import xarray as xr
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

SOILGRID_VARS = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "ocd", "sand", "silt", "soc"]

SOILGRID_DEPTH = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

SOILGRID_VALS = ["mean", "uncertainty", "Q0.05", "Q0.5", "Q0.95"]


def construct_soilgrid_layer(var, depth, val):
    return f"{var}_{depth}_{val}"


def download_layer(layer):

    #print(f"Doing {tile}, {layer}")

    # the destination file 
    datapath = Path('/Net/Groups/BGI/work_2/Landscapes_dynamics/downloads/soilgrids/Africa')
    datapath.mkdir(exist_ok=True, parents=True)
    dst_path = str(datapath/f'sg_africa_{layer}.tif')

    if Path(dst_path).is_file():
        try:
            rasterio.open(dst_path)
            return
        except:
            Path(dst_path).unlink()

    # create the downloading url in 3 steps
    sg_layer = f'{layer.split("_")[0]}/{layer}.vrt'
    location = f'https://files.isric.org/soilgrids/latest/data/{sg_layer}'
    sg_url = f'/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url={location}'

    done = False
    c = 0

    while not done:
        try:
            with rasterio.open(sg_url) as src:
                
                kwds = src.profile
                tags = src.tags() # Add soilgrids tags with creation info.
                kwds['driver'] = 'GTiff'
                kwds['tiled'] = True
                kwds['compress'] = 'deflate' # lzw or deflate
                kwds['dtype'] = 'int16' # soilgrids datatype
                kwds['nodata'] = -32768 # default nodata
                kwds['crs'] = CRS.from_epsg(4326)
                

                with WarpedVRT(src, crs=4326, resampling=Resampling.nearest) as vrt:

                    dst_window = vrt.window(-40, -25, 40, 55)

                    kwds['transform'] = vrt.window_transform(dst_window)

                    data = vrt.read(window=dst_window)

                    kwds['height'] = data.shape[1]
                    kwds['width'] = data.shape[2]
                    
                    with rasterio.open(dst_path, 'w', **kwds) as dst:
                        dst.update_tags(**tags)
                        dst.write(data)
            done = True
        except:
            print(f"Retrying {layer}")
            c+=1
            if c >= 5:
                done = True
            time.sleep(5)

    #print(f"Done {tile}, {layer}")



if __name__ == "__main__":


    # tiles = sorted([d.name for d in (Path("/Net/Groups/BGI/work_2/Landscapes_dynamics/downloads")/"Sentinel/FullEurope/tiles").iterdir()])

    layers = [construct_soilgrid_layer(var, depth, val) for var in SOILGRID_VARS for depth in SOILGRID_DEPTH for val in SOILGRID_VALS]

    # tiles_layers = [(tile, layer) for tile in tiles for layer in layers]

    # for tile_layer in tqdm(tiles_layers):
    #     download_tile_layer(tile_layer)

    for layer in tqdm(layers):
        print(f"Starting {layer}")
        download_layer(layer)

    # with ProcessPoolExecutor(max_workers=4) as pool:
    #     _ = list(tqdm(pool.map(download_layer, layers), total = len(layers)))

    # with ThreadPoolExecutor(max_workers=38) as pool:
    #     _ = list(tqdm(pool.map(download_tile_layer, tiles_layers), total = len(tiles_layers)))