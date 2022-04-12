
from . import provider_base, s2, sentinel1

PROVIDERS = {
    "s2": s2.sentinel2.Sentinel2,
    "s1": sentinel1.Sentinel1
}