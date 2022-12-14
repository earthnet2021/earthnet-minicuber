

import pystac_client
import stackstac
from rasterio import RasterioIOError
import os
import rasterio as rio
import argparse
import time


def time_stackstac(aws_bucket, use_gdal_session):

    if aws_bucket == "dea":
        URL = "https://explorer.digitalearth.africa/stac/"
        if not use_gdal_session:
            os.environ['AWS_S3_ENDPOINT'] = 's3.af-south-1.amazonaws.com'

    else:
        URL = "https://earth-search.aws.element84.com/v0"
        if not use_gdal_session:
            if 'AWS_S3_ENDPOINT' in os.environ:
                del os.environ['AWS_S3_ENDPOINT']

    gdal_session = stackstac.DEFAULT_GDAL_ENV.updated(always=dict(session=rio.session.AWSSession(aws_unsigned = True, endpoint_url = 's3.af-south-1.amazonaws.com' if aws_bucket == "dea" else None))) if use_gdal_session else None

    bbox = (-8., 31., -7.95, 31.05)

    catalog = pystac_client.Client.open(URL)
    search = catalog.search(
                bbox = bbox,
                collections=["s2_l2a" if aws_bucket == "dea" else "sentinel-s2-l2a-cogs"],
                datetime="2021-11-01/2021-11-15"
            )
    items_s2 = search.get_all_items()
    metadata = items_s2.to_dict()['features'][0]["properties"]
    epsg = metadata["proj:epsg"]

    if aws_bucket == "dea":
        with rio.Env(aws_unsigned = True, AWS_S3_ENDPOINT= 's3.af-south-1.amazonaws.com'):
            stack = stackstac.stack(items_s2, epsg = epsg, assets = ["B02", "B03", "B04"], dtype = "float32", band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 256,errors_as_nodata=(RasterioIOError('.*'), ), gdal_env=gdal_session)

            stack = stack.compute()
    else:
        stack = stackstac.stack(items_s2, epsg = epsg, assets = ["B02", "B03", "B04"], dtype = "float32", band_coords = False, bounds_latlon = bbox, xy_coords = 'center', chunksize = 256,errors_as_nodata=(RasterioIOError('.*'), ), gdal_env=gdal_session)

        stack = stack.compute()

    print(stack)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('bucket', type=str, help='dea or element84')
    parser.add_argument("--use_gdal_session", help="increase output verbosity",action="store_true")
    args = parser.parse_args()

    starttime = time.perf_counter()
    time_stackstac(aws_bucket = args.bucket, use_gdal_session = args.use_gdal_session)
    elapsed_time = time.perf_counter() - starttime
    print(f"Took {elapsed_time} seconds process time.")
    
