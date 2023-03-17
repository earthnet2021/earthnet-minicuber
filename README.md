# earthnet-minicuber

A Python library for creating EarthNet-style minicubes.

This library currently creates minicubes over Africa as they will be used in [DeepCube Horizon 2020](https://deepcube-h2020.eu/ "DeepCube Horizon 2020") project Use Case 1. It is developed by the [EarthNet](https://www.earthnet.tech/) Team.

It is currently under development, thus do expect bugs and please report them!


## Installation

Prerequisites (We use an Anaconda environment):

```
conda create -n minicuber python=3.10 gdal cartopy -c conda-forge
conda deactivate
conda activate minicuber
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scipy matplotlib seaborn netCDF4 xarray zarr dask shapely pillow pandas s3fs fsspec boto3 psycopg2 pystac-client stackstac planetary-computer rasterio[s3] rioxarray odc-algo segmentation-models-pytorch folium ipykernel ipywidgets
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

## Tutorial

See `notebooks/example.ipynb` for an usage example.