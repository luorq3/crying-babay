import re
import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def pre_handle(img0):
    img0 = img0.resize((224, 224), Image.BILINEAR)
    img0 = np.array(img0).astype('float32')
    img = img0.transpose((2, 0, 1)) / 255
    return img

def save_img(img_name, img, run_path):
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    img_path = os.path.join(run_path, img_name)
    img.save(img_path)
    print(f"Image saved to `{img_path}`.")
