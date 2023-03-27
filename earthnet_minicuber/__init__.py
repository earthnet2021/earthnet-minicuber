"""EarthNet Minicuber"""

__version__ = "0.1.3"
__author__ = "Vitus Benson"



from . import provider, minicuber, plot

from earthnet_minicuber.minicuber import Minicuber
from earthnet_minicuber.provider.provider_base import Provider
from earthnet_minicuber.provider import PROVIDERS
from earthnet_minicuber.plot import plot_rgb


load_minicube = Minicuber.load_minicube