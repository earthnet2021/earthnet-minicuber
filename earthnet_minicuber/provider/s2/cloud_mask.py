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


try: 
    import ee
except ImportError: 
    ee = None
import numpy as np


def PCL_s2cloudless(collection):
    """Creates a Potential Cloud Layer (PCL) using s2cloudless.

    Parameters
    ----------
    collection : ee.ImageColection
        Sentinel-2 L1C (TOA) collection to join.

    Returns
    -------
    ee.ImageCollection
        Collection with the PCL mask.
    """
    JOIN_NAME = "cloud_mask"
    THRESHOLD = 50
    S2CLOUDLESS = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    INDEX_FILTER = ee.Filter.equals(leftField="system:index", rightField="system:index")

    def threshold_s2cloudless(img):
        """Apply a threshold over the s2cloudless probabilities and creates a Potential Cloud Layer (PCL)"""

        clouds = ee.Image(img.get(JOIN_NAME)).select("probability")
        PCL = clouds.gte(THRESHOLD).rename("PCL")

        return img.addBands(PCL)

    S2WithCloudMask = ee.Join.saveFirst(JOIN_NAME).apply(
        collection, S2CLOUDLESS, INDEX_FILTER
    )
    S2WithCloudMask = ee.ImageCollection(S2WithCloudMask).map(threshold_s2cloudless)

    return S2WithCloudMask


def PCL_fmask(img):
    """Creates a Potential Cloud Layer (PCL) using Fmask version 3.2 (Zhu et al., 2015).

    Parameters
    ----------
    img : ee.Image
        Sentinel-2 L1C (TOA) image.

    Returns
    -------
    ee.Image
        Image with the PCL mask.
    """
    B = img.select("B2") / 10000
    G = img.select("B3") / 10000
    R = img.select("B4") / 10000
    N = img.select("B8") / 10000
    S1 = img.select("B11") / 10000
    S2 = img.select("B12") / 10000
    cirrus = img.select("B10") / 10000

    ##################################
    # Potential Cloud Layer (Pass one)
    ##################################

    NDSI = (G - S1) / (G + S1)
    NDVI = (N - R) / (N + R)

    ## Basic Test
    # Eq. 1. (Zhu and Woodcock, 2012)

    basic_test = (S2 > 0.03) & (NDSI < 0.8) & (NDVI < 0.8)  # There is no BT in S2

    ## Whiteness Test
    # Eq. 2. (Zhu and Woodcock, 2012)

    mean_vis = (B + G + R) / 3.0
    whiteness = (
        ((B - mean_vis) / mean_vis)
        + ((G - mean_vis) / mean_vis)
        + ((R - mean_vis) / mean_vis)
    ).abs()
    whiteness_test = whiteness < 0.7

    ## HOT Test
    # Eq. 3. (Zhu and Woodcock, 2012)

    HOT_test = (B - (0.5 * R) - 0.08) > 0

    ## B4/B5 Test
    # Eq. 4. (Zhu and Woodcock, 2012)

    B4_B5_test = N / S1 > 0.75

    ## Water Test
    # Eq. 5. (Zhu and Woodcock, 2012)

    water_test = ((NDVI < 0.01) & (N < 0.11)) | ((NDVI < 0.1) & (N < 0.05))

    ## Cirrus test expansion
    # Section 2.2.1. (Zhu et al., 2015)

    cirrus_test = cirrus > 0.01

    ## Potential Cloud Pixels
    # Eq. 6. (Zhu and Woodcock, 2012) + cirrus expansion from (Zhu et al., 2015)

    PCP = (basic_test & whiteness_test & HOT_test & B4_B5_test) | cirrus_test

    ##################################
    # Potential Cloud Layer (Pass two)
    ##################################

    # Eq. 10. (Zhu and Woodcock, 2012)
    brightness_prob = S1.min(0.11) / 0.11

    # Eq. 1. (Zhu et al., 2015)
    cirrus_prob = cirrus / 0.04

    # Eq. 2. (Zhu et al., 2015)
    wCloud_Prob = brightness_prob + cirrus_prob

    # Eq. 15. (Zhu and Woodcock, 2012)
    variability_prob = 1.0 - NDVI.abs().max(NDSI.abs()).max(whiteness)

    # Eq. 3. (Zhu et al., 2015)
    lCloud_Prob = variability_prob + cirrus_prob

    # Eq. 12. (Zhu and Woodcock, 2012)
    clear_sky_land = PCP.Not() & water_test.Not()

    # Eq. 17. (Zhu and Woodcock, 2012)
    # land_threshold = lCloud_Prob.updateMask(clear_sky_land).reduceRegion(ee.Reducer.percentile([82.5]),maxPixels = 1e12,bestEffort = True,scale = 30).getNumber("constant") + 0.2
    land_threshold = 0.5  # Original threshold

    # Dynamic water threshold (Section 2.1 -> Cloud Detection over Water) (Zhu et al., 2015)
    clear_sky_water = water_test & (S2 < 0.03)
    # water_threshold = wCloud_Prob.updateMask(clear_sky_water).reduceRegion(ee.Reducer.percentile([82.5]),maxPixels = 1e12,bestEffort = True,scale = 30).getNumber("constant") + 0.2
    water_threshold = 0.5  # Original threshold

    # Eq. 18. (Zhu and Woodcock, 2012) with dynamic water threshold and with removal of the 99% cloud probability test (Zhu et al., 2015)
    PCL = (PCP & water_test & (wCloud_Prob > water_threshold)) | (
        PCP & water_test.Not() & (lCloud_Prob > land_threshold)
    )  # | ((lCloud_Prob > 0.99) & water_test.Not())

    # "sets a pixel to cloud if five or more pixels in its 3-by-3 neighborhood are cloud pixels; otherwise, the pixel stays clear"
    PCL = ((PCL + PCL.focalMode(kernelType="square")) > 0).reproject(
        crs=img.select(0).projection(), scale=10
    )

    return img.addBands(PCL.rename("PCL"))


def PSL(img):
    """Creates a Potential Snow Layer (PSL) using Fmask.

    Parameters
    ----------
    img : ee.Image
        Sentinel-2 L1C (TOA) image.

    Returns
    -------
    ee.Image
        Image with the PSL mask.
    """
    G = img.select("B3") / 10000
    N = img.select("B8") / 10000
    S1 = img.select("B11") / 10000

    NDSI = (G - S1) / (G + S1)

    # Eq. 20. (Zhu and Woodcock, 2012)
    PSL = (NDSI > 0.15) & (N > 0.11) & (G > 0.1)

    return img.addBands(PSL.rename("PSL"))


def PCSL(img):
    """Creates a Potential Cloud Shadow Layer (PCSL) using Fmask version 3.2 (Zhu et al., 2015).

    Parameters
    ----------
    img : ee.Image
        Sentinel-2 L1C (TOA) image.

    Returns
    -------
    ee.Image
        Image with the PCSL mask.
    """
    N = img.select("B8") / 10000
    S1 = img.select("B11") / 10000

    # Eq. 19. (Zhu and Woodcock, 2012)
    flood_fill_N = ee.Algorithms.FMask.fillMinima((N * 10000).int16()) / 10000

    # SWIR band added (Section 2.1 -> Potential Shadow Detection) (Zhu et al., 2015)
    flood_fill_S1 = ee.Algorithms.FMask.fillMinima((S1 * 10000).int16()) / 10000

    PCSL = ((flood_fill_N - N) > 0.02) & ((flood_fill_S1 - S1) > 0.02)

    return img.addBands(PCSL.rename("PCSL"))


def matchShadows(img):
    """Matches a Potential Cloud Layer (PCL) with a Potential Cloud Shadow Layer (PCSL)
    using Fmask version 3.2 (Zhu et al., 2015).

    Parameters
    ----------
    img : ee.Image
        Sentinel-2 L1C (TOA) image.

    Returns
    -------
    ee.Image
        Image with the PCSL mask.
    """
    # Fixed values for S2 (Zhu et al., 2015)
    MIN_CLOUD_HEIGHT = 200
    MAX_CLOUD_HEIGHT = 1200
    DILATION = 8  # in pixels

    PCL = img.select("PCL")
    PCSL = img.select("PCSL")
    PSL = img.select("PSL")

    shadowAzimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # maxCloudDisplacement = MAX_CLOUD_HEIGHT / ((ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE")))) * (np.pi / 180)).tan()
    maxCloudDisplacement = ee.Number(1000)
    maxCloudProjection = (
        PCL.directionalDistanceTransform(
            shadowAzimuth, (maxCloudDisplacement / 10).round()
        )
        .reproject(crs=img.select(0).projection(), scale=10)
        .select("distance")
        .mask()
    )  # Pixel GSD of S2

    minCloudDisplacement = (
        MIN_CLOUD_HEIGHT
        / (
            (ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE"))))
            * (np.pi / 180)
        ).tan()
    )
    minCloudProjection = (
        PCL.directionalDistanceTransform(
            shadowAzimuth, (minCloudDisplacement / 10).round()
        )
        .reproject(crs=img.select(0).projection(), scale=10)
        .select("distance")
        .mask()
    )  # Pixel GSD of S2

    cloudProjection = maxCloudProjection - minCloudProjection

    # New Potential Cloud Shadow Layer
    PCSL = (
        (cloudProjection * PCSL)
        .focalMax(DILATION, kernelType="square")
        .reproject(crs=img.select(0).projection(), scale=10)
    )

    PCL = PCL.focalMax(DILATION, kernelType="square").reproject(
        crs=img.select(0).projection(), scale=10
    )  # 1: Clouds
    PCSL = PCSL * 2  # 2: Cloud Shadows
    PSL = (
        (PSL * 3)
        .focalMax(DILATION, kernelType="square")
        .reproject(crs=img.select(0).projection(), scale=10)
    )  # 3: Snow

    # Clouds are high priority, followed by cloud shadows and then snow
    mask = PSL.blend(PCSL.mask(PCSL)).blend(PCL.mask(PCL))

    return img.addBands(mask.rename("CLOUD_MASK"))


def fmask(img):
    """Computes the complete Fmask version 3.2 algorithm (Zhu et al., 2015).

    This algorithm is not used.

    Parameters
    ----------
    img : ee.Image
        Sentinel-2 L1C (TOA) image.

    Returns
    -------
    ee.Image
        Image with the Fmask version 3.2.
    """
    B = img.select("B2") / 10000
    G = img.select("B3") / 10000
    R = img.select("B4") / 10000
    N = img.select("B8") / 10000
    S1 = img.select("B11") / 10000
    S2 = img.select("B12") / 10000
    cirrus = img.select("B10") / 10000

    ##################################
    # Potential Cloud Layer (Pass one)
    ##################################

    NDSI = (G - S1) / (G + S1)
    NDVI = (N - R) / (N + R)

    ## Basic Test
    # Eq. 1. (Zhu and Woodcock, 2012)

    basic_test = (S2 > 0.03) & (NDSI < 0.8) & (NDVI < 0.8)  # There is no BT in S2

    ## Whiteness Test
    # Eq. 2. (Zhu and Woodcock, 2012)

    mean_vis = (B + G + R) / 3.0
    whiteness = (
        ((B - mean_vis) / mean_vis)
        + ((G - mean_vis) / mean_vis)
        + ((R - mean_vis) / mean_vis)
    ).abs()
    whiteness_test = whiteness < 0.7

    ## HOT Test
    # Eq. 3. (Zhu and Woodcock, 2012)

    HOT_test = (B - (0.5 * R) - 0.08) > 0

    ## B4/B5 Test
    # Eq. 4. (Zhu and Woodcock, 2012)

    B4_B5_test = N / S1 > 0.75

    ## Water Test
    # Eq. 5. (Zhu and Woodcock, 2012)

    water_test = ((NDVI < 0.01) & (N < 0.11)) | ((NDVI < 0.1) & (N < 0.05))

    ## Cirrus test expansion
    # Section 2.2.1. (Zhu et al., 2015)

    cirrus_test = cirrus > 0.01

    ## Potential Cloud Pixels
    # Eq. 6. (Zhu and Woodcock, 2012) + cirrus expansion from (Zhu et al., 2015)

    PCP = (basic_test & whiteness_test & HOT_test & B4_B5_test) | cirrus_test

    ##################################
    # Potential Cloud Layer (Pass two)
    ##################################

    # Eq. 10. (Zhu and Woodcock, 2012)
    brightness_prob = S1.min(0.11) / 0.11

    # Eq. 1. (Zhu et al., 2015)
    cirrus_prob = cirrus / 0.04

    # Eq. 2. (Zhu et al., 2015)
    wCloud_Prob = brightness_prob + cirrus_prob

    # Eq. 15. (Zhu and Woodcock, 2012)
    variability_prob = 1.0 - NDVI.abs().max(NDSI.abs()).max(whiteness)

    # Eq. 3. (Zhu et al., 2015)
    lCloud_Prob = variability_prob + cirrus_prob

    # Eq. 12. (Zhu and Woodcock, 2012)
    clear_sky_land = PCP.Not() & water_test.Not()

    # Eq. 17. (Zhu and Woodcock, 2012)
    # land_threshold = lCloud_Prob.updateMask(clear_sky_land).reduceRegion(ee.Reducer.percentile([82.5]),maxPixels = 1e12,bestEffort = True,scale = 30).getNumber("constant") + 0.2
    land_threshold = 0.5

    # Dynamic water threshold (Section 2.1 -> Cloud Detection over Water) (Zhu et al., 2015)
    clear_sky_water = water_test & (S2 < 0.03)
    # water_threshold = wCloud_Prob.updateMask(clear_sky_water).reduceRegion(ee.Reducer.percentile([82.5]),maxPixels = 1e12,bestEffort = True,scale = 30).getNumber("constant") + 0.2
    water_threshold = 0.5

    # Eq. 18. (Zhu and Woodcock, 2012) with dynamic water threshold and with removal of the 99% cloud probability test (Zhu et al., 2015)
    PCL = (PCP & water_test & (wCloud_Prob > water_threshold)) | (
        PCP & water_test.Not() & (lCloud_Prob > land_threshold)
    )  # | ((lCloud_Prob > 0.99) & water_test.Not())

    # "sets a pixel to cloud if five or more pixels in its 3-by-3 neighborhood are cloud pixels; otherwise, the pixel stays clear"
    PCL = ((PCL + PCL.focalMode(kernelType="square")) > 0).reproject(
        crs=img.select(0).projection(), scale=10
    )

    ##############################
    # Potential Cloud Shadow Layer
    ##############################

    # Eq. 19. (Zhu and Woodcock, 2012)
    flood_fill_N = ee.Algorithms.FMask.fillMinima((N * 10000).int16()) / 10000

    # SWIR band added (Section 2.1 -> Potential Shadow Detection) (Zhu et al., 2015)
    flood_fill_S1 = ee.Algorithms.FMask.fillMinima((S1 * 10000).int16()) / 10000

    PCSL = ((flood_fill_N - N) > 0.02) & ((flood_fill_S1 - S1) > 0.02)

    ######################
    # Potential Snow Layer
    ######################

    # Eq. 20. (Zhu and Woodcock, 2012)
    PSL = (NDSI > 0.15) & (N > 0.11) & (G > 0.1)

    ###########################################
    # Object-based cloud and cloud shadow match
    ###########################################

    # Fixed values for S2 (Zhu et al., 2015)
    MIN_CLOUD_HEIGHT = 200
    MAX_CLOUD_HEIGHT = 1200

    DILATION = 8  # in pixels

    shadowAzimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # maxCloudDisplacement = MAX_CLOUD_HEIGHT / ((ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE")))) * (np.pi / 180)).tan()
    maxCloudDisplacement = ee.Number(1000)
    maxCloudProjection = (
        PCL.directionalDistanceTransform(
            shadowAzimuth, (maxCloudDisplacement / 10).round()
        )
        .reproject(crs=img.select(0).projection(), scale=10)
        .select("distance")
        .mask()
    )  # Pixel GSD of S2

    minCloudDisplacement = (
        MIN_CLOUD_HEIGHT
        / (
            (ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_ZENITH_ANGLE"))))
            * (np.pi / 180)
        ).tan()
    )
    minCloudProjection = (
        PCL.directionalDistanceTransform(
            shadowAzimuth, (minCloudDisplacement / 10).round()
        )
        .reproject(crs=img.select(0).projection(), scale=10)
        .select("distance")
        .mask()
    )  # Pixel GSD of S2

    cloudProjection = maxCloudProjection - minCloudProjection

    # New Potential Cloud Shadow Layer
    PCSL = (
        (cloudProjection * PCSL)
        .focalMax(DILATION, kernelType="square")
        .reproject(crs=img.select(0).projection(), scale=10)
    )

    PCL = PCL.focalMax(DILATION, kernelType="square").reproject(
        crs=img.select(0).projection(), scale=10
    )  # 1: Clouds
    PCSL = PCSL * 2  # 2: Cloud Shadows
    PSL = (
        (PSL * 3)
        .focalMax(DILATION, kernelType="square")
        .reproject(crs=img.select(0).projection(), scale=10)
    )  # 3: Snow

    # Clouds are high priority, followed by cloud shadows and then snow
    fmask_image = PSL.blend(PCSL.mask(PCSL)).blend(PCL.mask(PCL))

    return img.addBands(fmask_image.rename("FMASK"))
