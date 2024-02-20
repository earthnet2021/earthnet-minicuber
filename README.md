
# EarthNet Minicuber

*A Python library for creating EarthNet-style minicubes.*


<a href='https://pypi.python.org/pypi/earthnet-minicuber'>
    <img src='https://img.shields.io/pypi/v/earthnet-minicuber.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://twitter.com/vitusbenson" target="_blank">
    <img src="https://img.shields.io/twitter/follow/vitusbenson?style=social" alt="Twitter">
</a>


**GitHub**: [https://github.com/earthnet2021/earthnet-minicuber](https://github.com/earthnet2021/earthnet-minicuber)

**PyPI**: [https://pypi.org/project/earthnet-minicuber/](https://pypi.org/project/earthnet-minicuber/)



This package creates minicubes from cloud storage using STAC catalogues. A minicube usually contains a satellite image time series of Sentinel 2 imagery alongside other complementary information, all re-gridded to a common grid. This package implements a cloud mask based on deep learning, which allows for analysis-ready Sentinel 2 imagery.

It is currently under development, thus do expect bugs and please report them!


## Tutorial

1. Loading the package
```Python
import earthnet_minicuber as emc
```

2. Creating a dictionary with specifications of the desired minicube
```Python
specs = {
    "lon_lat": (43.598946, 3.087414), # center pixel
    "xy_shape": (256, 256), # width, height of cutout around center pixel
    "resolution": 10, # in meters.. will use this on a local UTM grid..
    "time_interval": "2021-07-01/2021-07-31",
    "providers": [
        {
            "name": "s2",
            "kwargs": {"bands": ["B02", "B03", "B04", "B8A"], "best_orbit_filter": True, "five_daily_filter": False, "brdf_correction": True, "cloud_mask": True, "aws_bucket": "planetary_computer"}
        },
        {
            "name": "s1",
            "kwargs": {"bands": ["vv", "vh"], "speckle_filter": True, "speckle_filter_kwargs": {"type": "lee", "size": 9}, "aws_bucket": "planetary_computer"} 
        },
        {
            "name": "ndviclim",
            "kwargs": {"bands": ["mean", "std"]}
        },
        {
            "name": "cop",
            "kwargs": {}
        },
        {
            "name": "esawc",
            "kwargs": {"bands": ["lc"], "aws_bucket": "planetary_computer"}
        }
        ]
}
```

3. Downloading the minicube
```Python
mc = emc.load_minicube(specs, compute = True)
```

4. Plotting cloud-masked Sentinel 2 RGB imagery
```Python
emc.plot_rgb(mc)
```

See `notebooks/example.ipynb` for a more detailed usage example.



## Data Providers

The minicuber is centered around the concept of data providers, which wrap a data source and handle data loading of that source. The `emc.Minicuber` class then manages these data providers, by telling them the spatio-temporal range for which data needs to be loaded and afterwards re-gridding all data to a common reference frame (UTM grid).

### Sentinel 2

The Sentinel 2 provider loads and processes Copernicus Sentinel 2 imagery.

Kwargs:
- `bands`: choose any subset from `["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "WVP", "AOT", "SCL"]`.
- `aws_bucket`: We currently support data loading from three cloud buckets: Microsoft Planetary Computer (`"planetary_computer"`), Element84 AWS bucket (`element84`) and DigitalEarthAfrica AWS bucket (`dea`). We recommend using the Microsoft planetary computer with the keyword argument `aws_bucket = "planetary_computer"`.
- `best_orbit_filter`: Sentinel 2 has a regular overpass frequency of 5 days. However, sometimes it can be smaller due to off-nadir captures. Such captures change the viewing angle of the scene. If `True`, this filter finds the best orbit and then only returns imagery from a regular 5-daily cycle.
- `five_daily_filter`: If `True` returns a regular 5-daily cycle starting with the first date in `full_time_interval`. It has no effect, if `best_orbit_filter` is used.
- `brdf_correction`: If `True`, does BRDF correction based on the Sentinel 2 Metadata (illumination angles).
- `cloud_mask`: If `True`, creates a cloud and cloud shadow mask based on deep learning. It automatically finds the best available cloud mask for the requested `bands`.
- `cloud_mask_rescale_factor`: If using cloud mask and a lower resolution than 10m, set this rescaling factor to the multiple of 10m that you are requesting. E.g. if `resolution = 20`, set `cloud_mask_rescale_factor = 2`.
- `correct_processing_baseline`: If `True` (default): corrects the shift of +1000 that exists in Sentinel 2 data with processing baseline >= 4.0


## Installation

Prerequisites (We use an Anaconda environment):

```
conda create -n minicuber python=3.10 gdal cartopy -c conda-forge
conda deactivate
conda activate minicuber
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scipy matplotlib seaborn netCDF4 xarray zarr dask shapely pillow pandas s3fs fsspec boto3 psycopg2 pystac-client stackstac planetary-computer rasterio rioxarray odc-algo segmentation-models-pytorch folium ipykernel ipywidgets sen2nbar
```

Install this package with PyPI:
```
pip install earthnet-minicuber
```

or install this package in developing mode with
```
git clone https://github.com/earthnet2021/earthnet-minicuber.git
cd earthnet-minicuber
pip install -e .
```

or directly with
```
pip install git+https://github.com/earthnet2021/earthnet-minicuber.git
```

## Similar Packages

This package is build on top of [stackstac](https://stackstac.readthedocs.io/en/latest/), which allows accessing data stored in cloud-optimized geotiffs with xarray.

Similar to this package, [cubo](https://github.com/davemlz/cubo) provides a high-level interface to stackstac.


## Acknowledgement

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101004188 ([DeepCube Horizon 2020](https://deepcube-h2020.eu/ "DeepCube Horizon 2020")). We are grateful to David Montero Loaiza for providing the [sen2nbar](https://github.com/ESDS-Leipzig/sen2nbar) package used for the Sentinel 2 BRDF correction. We are grateful to César Aybar and the [CloudSEN12](https://cloudsen12.github.io/) team, their work forms the basis for the cloud mask implemented in earthnet-minicuber.
