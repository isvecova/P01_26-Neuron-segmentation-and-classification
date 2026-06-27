import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
from natsort import natsorted

io.logger_setup() # run this to get printing of progress


model = models.CellposeModel(gpu=True)

# *** change to your google drive folder path ***
dir = Path(r"D:\OneDrive - IEM\000_inbox\260604_Kralova_python_troubleshooting\cellpose + far-red channel\train")
if not dir.exists():
  raise FileNotFoundError("directory does not exist")

# *** change to your image extension ***
image_ext = ".tif"

# list all files
files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])

if(len(files)==0):
  raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
else:
  print(f"{len(files)} images in folder:")

for f in files:
  print(f.name)

CELLPROB_THRESHOLD = -5.0
  
masks_ext = ".tif"
for i in trange(len(files)):
    f = files[i]
    img = io.imread(f)
    masks, flows, styles = model.eval(img, cellprob_threshold=CELLPROB_THRESHOLD)
    io.imsave(dir / (f.stem + "_masks" + masks_ext), masks)
    io.masks_flows_to_seg(img, masks, flows, f, channels=None, imgs_restore=None, restore_type=None, ratio=1.0)

