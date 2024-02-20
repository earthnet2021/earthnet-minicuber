from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "torch==1.13.1",
    "timm",
    "torchvision",
    "segmentation-models-pytorch",
    "numpy",
    "matplotlib",
    "pillow",
    "xarray",
    "zarr",
    "dask",
    "netcdf4",
    "pandas",
    "planetary-computer",
    "pyproj",
    "pystac-client",
    "rasterio",
    "requests",
    "stackstac",
    "rioxarray",
    "shapely",
    "fsspec",
    "aiohttp",
    "odc-algo",
    "sen2nbar",
    "s3fs"
    ]


setup(name='earthnet-minicuber', 
        version='0.1.3',
        description="EarthNet Minicuber",
        author="Vitus Benson, Christian Requena-Mesa",
        author_email="vbenson@bgc-jena.mpg.de",
        url="https://earthnet.tech",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3"
                 ],
        packages=['earthnet_minicuber', 'earthnet_minicuber.provider', 'earthnet_minicuber.provider.s2'],#find_packages(),
        install_requires=install_requires,
        extras_require={
            "EE": ["earthengine-api","wxee","eemont"],
        }
        )
