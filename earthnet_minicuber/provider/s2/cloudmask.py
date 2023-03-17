
import numpy as np
import segmentation_models_pytorch as smp 
import xarray as xr

import torch

from torch.utils.model_zoo import load_url

def get_checkpoint(bands_avail):

    bands_avail = set(bands_avail)

    if set(['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12', 'AOT', 'WVP']).issubset(bands_avail):
        ckpt = load_url("https://nextcloud.bgc-jena.mpg.de/s/qHKcyZpzHtXnzL2/download/mobilenetv2_l2a_all.pth")
        ckpt_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12', 'AOT', 'WVP']
    elif set(["B02", "B03", "B04", "B8A"]).issubset(bands_avail):
        ckpt = load_url("https://nextcloud.bgc-jena.mpg.de/s/Ti4aYdHe2m3jBHy/download/mobilenetv2_l2a_rgbnir.pth")
        ckpt_bands = ["B02", "B03", "B04", "B8A"]
    else:
        raise Exception(f"The bands {bands_avail} do not contain the necessary bands for cloud masking. Please include at least bands B02, B03, B04 and B8A.")
        ckpt = None
        ckpt_bands = None

    return ckpt, ckpt_bands


class CloudMask:
    def __init__(self, bands = ["B02", "B03", "B04", "B8A"]):

        self.bands = bands
        ckpt, self.ckpt_bands = get_checkpoint(bands)

        self.model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                classes=4,
                in_channels=len(self.ckpt_bands)       
        )

        if ckpt:
            self.model.load_state_dict(ckpt)

        self.model.eval()

        self.bands_scale = xr.DataArray(12*[10000,] + [65535, 65535, 1], coords = {"band": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "AOT", "WVP", "SCL"]})

    def __call__(self, stack):

        ds = stack.to_dataset("band")

        x = torch.from_numpy((stack.sel(band = self.ckpt_bands)/self.bands_scale).fillna(1.0).transpose("time", "band", "y", "x").values.astype("float32"))

        b, c, h, w = x.shape

        h_big = ((h//32 + 1)*32)
        h_pad_left = (h_big - h)//2
        h_pad_right = ((h_big - h) + 1)//2

        w_big = ((w//32 + 1)*32)
        w_pad_left = (w_big - w)//2
        w_pad_right = ((w_big - w) + 1)//2

        x = torch.nn.functional.pad(x, (w_pad_left, w_pad_right, h_pad_left, h_pad_right), mode = "reflect")

        with torch.no_grad():
            y_hat = self.model(x)

        y_hat = y_hat[:, :, h_pad_left:-h_pad_right, w_pad_left:-w_pad_right]

        ds["mask"] = (("time", "y", "x"),torch.argmax(y_hat, dim = 1).float().cpu().numpy())

        return ds.to_array("band")
    
def cloud_mask_reduce(x, axis = None, **kwargs):
    return np.where((x==1).any(axis = axis), 1, np.where((x==3).any(axis = axis), 3, np.where((x==2).any(axis = axis), 2, np.where((x==0).any(axis = axis), 0, 4))))
