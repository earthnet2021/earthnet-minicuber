# earthnet-minicuber

A Python library for creating EarthNet-style minicubes.

This library currently creates minicubes over Africa as they will be used in [DeepCube Horizon 2020](https://deepcube-h2020.eu/ "DeepCube Horizon 2020") project Use Case 1. It is developed by the [EarthNet](https://www.earthnet.tech/) Team.

It is currently under development, thus do expect bugs and please report them!


## Installation

Prerequisites (We use an Anaconda environment):

```
conda create -n minicuber python=3.9
conda deactivate
conda activate minicuber
conda install -c conda-forge mamba
mamba install -c conda-forge numpy scipy matplotlib cartopy seaborn netCDF4 xarray zarr dask shapely rasterio rioxarray pillow pandas shapely
pip install pystac_client stackstac earthengine-api eemont planetary_computer folium cdsapi wxee
```

Install this package in developing mode with
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