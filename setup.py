from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "numpy",
    "matplotlib",
    "pillow",
    "xarray",
    "zarr",
    "dask",
    "netcdf4",
    "earthnet",
    "earthengine-api",
    "eemont",
    "pandas",
    "planetary_computer",
    "pyproj",
    "pystac_client",
    "rasterio",
    "requests",
    "stackstac",
    "wxee",
    "rioxarray",
    "shapely",
    "fsspec",
    "aiohttp"
    ]


setup(name='earthnet-minicuber', 
        version='0.0.1',
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
        packages=find_packages(),
        install_requires=install_requires,
        )
