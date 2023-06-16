import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
from torch.nn import functional as F
from torchvision import models
import torch.nn as nn
import torch
import random
import PIL

def ImageClass():
    pass

def CleaningImage(Dir:str):
    images = []
    label = []
    import os
    for dirname, _, filenames in os.walk(Dir):
        for filename in filenames:
            images.append(os.path.join(dirname, filename).split('\\')[-1])
            label.append(os.path.join(dirname, filename).split('\\')[-2])

    labels = np.unique(label)
    ints = np.arange(0, len(label))
    dicts = dict(zip(labels, ints))
    print(dicts)


    df_full = pd.DataFrame({'image_id': images, 'label': label})

    # FINDING BAD FILE

    index = []
    for i in range(len(df_full)):
        try:
            Image.open(PATH + str(df_full['label'].values[i]) + '\\' + str(df_full['image_id'].values[i]))

        except PIL.UnidentifiedImageError:
            index.append(i)

    df = df_full.iloc[index]
    print(df)

    pass


if __name__ == '__main__':
    PATH = "PYTHON_TO_IMANGENET/train/"
    CleaningImage(PATH)
    pass