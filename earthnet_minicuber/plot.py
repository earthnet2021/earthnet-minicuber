
import numpy as np

def plot_rgb(mc, mask = True):

    mc = mc.sel(time = mc.s2_avail == 1)
    
    if mask:
        return mc[["s2_B04", "s2_B03", "s2_B02"]].to_array("band").where(((mc.s2_mask < 1) & mc.s2_SCL.isin([1,2,4,5,6,7]))).plot.imshow(col = "time", col_wrap = 3, vmin = 0.0, vmax = 0.4)
    else:
        return mc[["s2_B04", "s2_B03", "s2_B02"]].to_array("band").plot.imshow(col = "time", col_wrap = 3, vmin = 0.0, vmax = 0.4)