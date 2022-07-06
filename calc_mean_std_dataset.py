import imp
from pathlib import Path
from tkinter import image_names
from PIL import Image
from kornia import total_variation
import numpy as np
from tqdm import tqdm
from IPython import embed

normal_set = Path("/DATA2/gaoha/liumd/sml/sml/selfgen/selfgen/normal")

total_images = 0
shape = None

for scene in normal_set.iterdir():
    rgb_v = scene / "1" / "rgb_v"
    total_images += len(list(rgb_v.iterdir()))
    if shape is None:
        img_file = next(rgb_v.iterdir())
        img = Image.open(img_file)
        img = np.asarray(img)
        shape = img.shape

print(total_images, "in total")
imgs = np.zeros((total_images, *shape))
ptr = 0
print(total_images, "in total, with total_image shape", imgs.shape)

for scene in normal_set.iterdir():
    rgb_v = scene / "1" / "rgb_v"
    for img_file in tqdm(list(rgb_v.iterdir())):
        img = Image.open(img_file)
        img = np.asarray(img)
        imgs[ptr] = img
        ptr += 1

print("mean_std", (
    imgs.mean(axis=(0,1,2)) / 255,
    imgs.std(axis=(0,1,2)) / 255
))

