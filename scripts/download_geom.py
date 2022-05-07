
import urllib.request
from tqdm import tqdm
import tarfile
import os
from pathlib import Path
from minio import Minio
import sys

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

vals = ["geom"]#["aspect-cosine", "aspect-sine", "aspect", "convergence", "cti", "dev-magnitude", "dev-scale", "dx", "dxx", "dxy", "dy", "dyy", "eastness", "elev-stdev", "geom", "northness", "pcurv", "rough-magnitude", "rough-scale", "roughness", "slope", "spi", "tcurv", "tpi", "tri", "vrm"]

base = "https://opentopography.s3.sdsc.edu/minio/download/dataspace/OTDS.012020.4326.1/raster/"
alt_base = "https://cloud.sdsc.edu/v1/AUTH_opentopography/hosted_data/OTDS.012020.4326.1/raster/"

tiles = ['s30w030', 's60w030'] #[f"{lon}{lat}" for lon in ["w030, e000, e030"] for lat in ["s60", "s30","n00", "n30"]]
# ["n30e030","n00w030","n00e000","n00e030","s30e000","s30e030","s60e000","s60e030"] "n30w030","n60w030","n30e000","n60e000",

targ_paths = [alt_base+f"{v}/{v}_90M_{t}.tar.gz" for v in vals for t in tiles]

outpath = Path("/Net/Groups/BGI/work_2/Landscapes_dynamics/downloads/Geomorphons/")

client = Minio("opentopography.s3.sdsc.edu")

for idx, targ_path in enumerate(sorted(targ_paths)):
    val, filename = targ_path.split("/")[-2:]
    print(f"{idx} of {len(targ_paths)} -- {filename}")
    val_path = outpath/val
    val_path.mkdir(parents = True, exist_ok = True)
    tmp_path = val_path/filename
    #client.fget_object("dataspace","/".join(targ_path.split("/")[-4:]),str(tmp_path))
    data = client.get_object("dataspace", "/".join(targ_path.split("/")[-4:]))
    with open(tmp_path, 'wb') as file_data, tqdm(unit="B", unit_scale=True, unit_divisor=1024, total = int(data.info()["Content-Length"]), file=sys.stdout, desc=filename) as progress:
        for d in data.stream(32*1024):
            datasize = file_data.write(d)
            progress.update(datasize)
    data.close()
    # done = False
    # while not done:
    #     try:
    #         with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
    #             urllib.request.urlretrieve(targ_path, filename = tmp_path, reporthook=t.update_to)
    #         done = True
    #     except:
    #         done = False
    with tarfile.open(tmp_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(iterable=members, total=len(members)):
            tar.extract(member=member,path=val_path)
    os.remove(tmp_path)


    # AFTERWARDS DO FROM TARGET DIRECTORY:
    # rio merge *.tif geom_90M_africa_europe.tif --co tiled=true --co blockxsize=256 --co blockysize=256