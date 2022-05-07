
from . import provider_base, s2, sentinel1, ndviclim, srtm, esawc, era5, soilgrids, geomorphons, landsat, cop30, alos

PROVIDERS = {
    "s2": s2.sentinel2.Sentinel2,
    "s1": sentinel1.Sentinel1,
    "ndviclim": ndviclim.NDVIClim,
    "srtm": srtm.SRTM,
    "esawc": esawc.ESAWorldcover,
    "era5": era5.ERA5,
    "sg": soilgrids.Soilgrids,
    "geom": geomorphons.Geomorphons,
    "ls": landsat.Landsat,
    "cop": cop30.Copernicus30,
    "alos": alos.ALOSWorld
}