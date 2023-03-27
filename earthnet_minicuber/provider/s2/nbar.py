
import numpy as np
import xarray as xr

from sen2nbar.c_factor import c_factor_from_item

def correct_processing_baseline(stack, items):
    """
    Adapted from https://github.com/ESDS-Leipzig/sen2nbar/blob/main/sen2nbar/nbar.py#L105 
    """
    items_dict = {item.id: item for item in items}
    ordered_items = [items_dict[itemid] for itemid in stack.id.values]
    processing_baseline = []
    for item in ordered_items:
        processing_baseline.append(item.properties["s2:processing_baseline"])
    
    # Processing baseline as data array
    processing_baseline = xr.DataArray(
        [float(x) for x in processing_baseline],
        dims="time",
        coords=dict(time=stack.time.values),
    )

    # Whether to shift the DN values
    # After 04.00 all DN values are shifted by 1000
    harmonize = xr.where(processing_baseline >= 4.0, -1000, 0)

    orig_bands = stack.band.values.tolist()

    stack_to_harmonize = stack.to_dataset("band")[[v for v in orig_bands if v in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]]].to_array("band")

    stack_rest = stack.to_dataset("band")[[v for v in orig_bands if v not in stack_to_harmonize.band]]

    # Zeros are NaN
    stack_to_harmonize = stack_to_harmonize.where(lambda x: x > 0, other=np.nan)

    # Harmonize (use processing baseline)
    stack_harmonized = stack_to_harmonize + harmonize
    
    stack = xr.merge([stack_harmonized.to_dataset("band"), stack_rest])[orig_bands].to_array("band")

    return stack


def call_sen2nbar(stack, items, epsg):
    """
    Adapted from https://github.com/ESDS-Leipzig/sen2nbar/blob/main/sen2nbar/nbar.py#L105 
    """

    items_dict = {item.id: item for item in items}
    ordered_items = [items_dict[itemid] for itemid in stack.id.values]

    # Compute the c-factor per item and extract the processing baseline
    c_array = []
    for item in ordered_items:#tqdm(ordered_items, disable=quiet):
        try:
            c = c_factor_from_item(item, f"epsg:{epsg}")
            c = c.interp(
                y=stack.y.values,
                x=stack.x.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        except ValueError:
            c = xr.DataArray(np.full((9,len(stack.y), len(stack.x)), np.NaN), coords = {"band": ['B02','B03','B04','B05','B06','B07','B08','B11','B12'], "y": stack.y, "x": stack.x}, dims = ("band", "y", "x"))
        c_array.append(c)

    orig_bands = stack.band.values.tolist()

    # Concatenate c-factor
    c = xr.concat(c_array, dim="time")
    c["time"] = stack.time.values

    stack_to_nbar = stack.to_dataset("band")[[v for v in c.to_dataset("band").data_vars.keys() if v in stack.band]].to_array("band")

    stack_rest = stack.to_dataset("band")[[v for v in stack.to_dataset("band").data_vars.keys() if v not in stack_to_nbar.band]]

    # Compute NBAR
    stack_nbar = stack_to_nbar * c

    stack_out = xr.merge([stack_nbar.to_dataset("band"), stack_rest])[orig_bands].to_array("band")

    return stack_out