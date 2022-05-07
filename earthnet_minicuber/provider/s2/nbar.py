"""
This code is originally Core code for the sen2flux dataset from https://github.com/davemlz/sen2flux. It is licensed under:

The MIT License (MIT)

Copyright (c) 2022 David Montero Loaiza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import xarray as xr


def kgeo(sunZenith, viewZenith, relativeAzimuth, br=1.0, hb=2.0):
    """Computes the Geometric Kernel (Kgeo).

    Parameters
    ----------
    sunZenith : xarray.DataArray
        Sun Zenith angles in degrees.
    viewZenith : xarray.DataArray
        Sensor Zenith angles in degrees.
    relativeAzimuth : xarray.DataArray
        Relative Azimuth angles in degrees.
    br : float | int, default = 1.0
        Br factor.
    hb : float | int, default = 2.0
        Hb factor.

    Returns
    -------
    xarray.DataArray
        Geometric Kernel (Kgeo).
    """
    theta_i = np.deg2rad(sunZenith)
    theta_v = np.deg2rad(viewZenith)

    phi = np.deg2rad(relativeAzimuth)

    theta_i_dev = np.arctan(br * np.tan(theta_i))
    theta_v_dev = np.arctan(br * np.tan(theta_v))

    cos_xi_dev = np.cos(theta_i_dev) * np.cos(theta_v_dev) + np.sin(
        theta_i_dev
    ) * np.sin(theta_v_dev) * np.cos(phi)

    D = np.sqrt(
        (np.tan(theta_i_dev) ** 2.0)
        + (np.tan(theta_v_dev) ** 2.0)
        - (2.0 * np.tan(theta_i_dev) * np.tan(theta_v_dev) * np.cos(phi))
    )

    cos_t = (
        hb
        * np.sqrt(
            (D**2.0)
            + ((np.tan(theta_i_dev) * np.tan(theta_v_dev) * np.sin(phi)) ** 2.0)
        )
        / ((1.0 / np.cos(theta_i_dev)) + (1.0 / np.cos(theta_v_dev)))
    )

    # cos_t[cos_t > 1.0] = 1.0
    # cos_t[cos_t < -1.0] = -1.0

    cos_t = cos_t.where(lambda x: x <= 1.0, other=1.0)
    cos_t = cos_t.where(lambda x: x >= -1.0, other=-1.0)

    t = np.arccos(cos_t)

    O = (
        (1.0 / np.pi)
        * (t - np.sin(t) * cos_t)
        * ((1.0 / np.cos(theta_i_dev)) + (1.0 / np.cos(theta_v_dev)))
    )

    return (
        O
        - (1.0 / np.cos(theta_i_dev))
        - (1.0 / np.cos(theta_v_dev))
        + 0.5
        * (1.0 + cos_xi_dev)
        * (1.0 / np.cos(theta_i_dev))
        * (1.0 / np.cos(theta_v_dev))
    )


def kvol(sunZenith, viewZenith, relativeAzimuth):
    """Computes the Volumetric Kernel (Kvol).

    Parameters
    ----------
    sunZenith : xarray.DataArray
        Sun Zenith angles in degrees.
    viewZenith : xarray.DataArray
        Sensor Zenith angles in degrees.
    relativeAzimuth : xarray.DataArray
        Relative Azimuth angles in degrees.

    Returns
    -------
    xarray.DataArray
        Volumeric Kernel (Kvol).
    """
    theta_i = np.deg2rad(sunZenith)
    theta_v = np.deg2rad(viewZenith)

    phi = np.deg2rad(relativeAzimuth)

    cos_xi = np.cos(theta_i) * np.cos(theta_v) + np.sin(theta_i) * np.sin(
        theta_v
    ) * np.cos(phi)

    xi = np.arccos(cos_xi)

    return (
        (((np.pi / 2.0) - xi) * cos_xi + np.sin(xi))
        / (np.cos(theta_i) + np.cos(theta_v))
    ) - (np.pi / 4.0)


def brdf(fiso, fvol, fgeo, sunZenith, viewZenith, relativeAzimuth):
    """Computes the Bidirectional Reflectance Distribution Function (BRDF).

    Parameters
    ----------
    fiso : list
        Isometric parameters for each band.
    fvol : list
        Volumetric parameters for each band.
    fgeo : list
        Geometric parameters for each band.
    sunZenith : xarray.DataArray
        Sun Zenith angles in degrees.
    viewZenith : xarray.DataArray
        Sensor Zenith angles in degrees.
    relativeAzimuth : xarray.DataArray
        Relative Azimuth angles in degrees.

    Returns
    -------
    xarray.DataArray
        BRDF.
    """
    bands_interim = sunZenith.band.values
    # [
    #     "B02_Interim",
    #     "B03_Interim",
    #     "B04_Interim",
    #     "B05_Interim",
    #     "B06_Interim",
    #     "B07_Interim",
    #     "B08_Interim",
    #     "B11_Interim",
    #     "B12_Interim",
    # ]

    kvol_ = kvol(sunZenith, viewZenith, relativeAzimuth)
    fvol_kvol = xr.concat(
        [fvol[i] * kvol_[i] for i in range(len(fvol))], dim="band"
    ).assign_coords(band=("band", bands_interim))

    kgeo_ = kgeo(sunZenith, viewZenith, relativeAzimuth)
    fgeo_kgeo = xr.concat(
        [fgeo[i] * kgeo_[i] for i in range(len(fgeo))], dim="band"
    ).assign_coords(band=("band", bands_interim))

    kvol_kgeo = fvol_kvol + fgeo_kgeo

    brdf_ = xr.concat(
        [fiso[i] + kvol_kgeo[i] for i in range(len(fiso))], dim="band"
    ).assign_coords(band=("band", bands_interim))

    return brdf_


def c_factor(fiso, fvol, fgeo, sunZenith, viewZenith, relativeAzimuth):
    """Computes the c-factor.

    Parameters
    ----------
    fiso : list
        Isometric parameters for each band.
    fvol : list
        Volumetric parameters for each band.
    fgeo : list
        Geometric parameters for each band.
    sunZenith : xarray.DataArray
        Sun Zenith angles in degrees.
    viewZenith : xarray.DataArray
        Sensor Zenith angles in degrees.
    relativeAzimuth : xarray.DataArray
        Relative Azimuth angles in degrees.

    Returns
    -------
    xarray.DataArray
        c-factor.
    """
    return brdf(fiso, fvol, fgeo, sunZenith, viewZenith * 0, relativeAzimuth) / brdf(
        fiso, fvol, fgeo, sunZenith, viewZenith, relativeAzimuth
    )


def nbar(x):
    """Computes the Nadir BRDF Adjusted Reflectance (NBAR).

    Parameters
    ----------
    x : xarray.DataArray
        Single array in a DataArray.

    Returns
    -------
    xarray.DataArray
        NBAR.
    """

    bands = [b for b in x.band.values if b in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]]

    bands_idxs = [i for i, b in enumerate(["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]) if b in bands]

    bands_interim = [f"{b}_Interim" for b in bands]

    bands_nbar = [f"{b}_NBAR" for b in bands]


    SunAzimuth = x.sel(band="Sun_Azimuth")
    SunZenith = x.sel(band="Sun_Zenith")

    reflectance = x.sel(
        band=bands
    )

    reflectance = reflectance.assign_coords(band=("band", bands_interim)) / 10000

    ViewAzimuth = x.sel(
        band= [f"{b}_Azimuth" for b in bands]
    )

    ViewZenith = x.sel(
        band= [f"{b}_Zenith" for b in bands]
    )

    DeltaAzimuth = xr.concat(
        [SunAzimuth - i for i in ViewAzimuth], dim="band"
    ).assign_coords(band=("band", bands_interim))

    theta = xr.concat([SunZenith for i in range(len(bands))], dim="band").assign_coords(
        band=("band", bands_interim)
    )
    vartheta = ViewZenith.assign_coords(band=("band", bands_interim))
    phi = DeltaAzimuth.assign_coords(band=("band", bands_interim))

    fiso = [[0.0774, 0.1306, 0.1690, 0.2085, 0.2316, 0.2599, 0.3093, 0.3430, 0.2658][i] for i in bands_idxs]
    fgeo = [[0.0079, 0.0178, 0.0227, 0.0256, 0.0273, 0.0294, 0.0330, 0.0453, 0.0387][i] for i in bands_idxs]
    fvol = [[0.0372, 0.0580, 0.0574, 0.0845, 0.1003, 0.1197, 0.1535, 0.1154, 0.0639][i] for i in bands_idxs]

    nbar = c_factor(fiso, fvol, fgeo, theta, vartheta, phi) * reflectance

    return nbar.assign_coords(band=("band", bands_nbar))
